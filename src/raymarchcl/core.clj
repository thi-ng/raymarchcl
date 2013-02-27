(ns raymarchcl.core
  (:require
   [simplecl.core :as cl]
   [simplecl.utils :as clu]
   [simplecl.ops :as ops]
   [structgen.core :as sg]
   [structgen.parser :as sp]
   [piksel.core :as pix]))

(def cl-program
  "Location of the OpenCL program"
  "renderer.cl")

;; parse typedefs in OpenCL program and register all found struct types
(sg/reset-registry!)
(sg/register! (sp/parse-specs (slurp (clu/resource-stream cl-program))))

(defn render-options
  [{:keys [width height vres t]}]
  (let [d 0.015]
    {:resolution [width height],
     :flareAmp 0.01,
     :materials
     [{:albedo [0.9 0.9 0.9 1.0], :r0 0.5, :smoothness 0.9}
      {:albedo [1.9 0.9 0.9 1.0], :r0 0.4, :smoothness 0.9}],
     :maxDist 30,
     :invAspect (float (/ height width)),
     :eps 0.005,
     :skyColor [0.8 0.8 0.8],
     :startDist 0.0,
     :isoVal 0.125,
     :voxelRes vres
     :maxVoxelIter 80,
     :lightPos [0.0 2.0 0.0],
     :shadowBias 0.1,
     :aoAmp 0.2,
     :voxelBoundsMin [-0.97 -0.97 -0.97],
     :aoStepDist 0.2,
     :eyePos [1 2 2],
     :aoIter 5,
     :voxelSize 0.05,
     :normOffsets [[d (- d) (- d) 0] [(- d) (- d) d 0] [(- d) d (- d) 0] [d d d 0]],
     :frameBlend 0.05,
     :groundY 1.5,
     :shadowIter 48,
     :lightColor [50 50 50],
     :targetPos [0 0 0],
     :maxIter 64,
     :dof 0.05,
     :exposure 3.5,
     :minLightAtt 0.0,
     :voxelBounds2 [2 2 2],
     :time t,
     :fogDensity 0.025,
     :voxelBoundsMax [0.97 0.97 0.97],
     :lightScatter 0.05,
     :invVoxelScale [0.5 0.5 0.5],
     :up [0 1 0],
     :voxelBounds [1 1 1],
     :gamma 1.5}))

(defn gyroid [s t p]
  "Evaluates gyroid function at scaled point `v`."
  (let [[x y z] (map #(* s %) p)]
    (- (Math/abs
        (+ (* (Math/cos x) (Math/sin z))
           (* (Math/cos y) (Math/sin x))
           (* (Math/cos z) (Math/sin y))))
       t)))

(defn make-volume
  [{[rx ry rz] :vres}]
  (prn "volume " rx ry rz)
  (let [voxels (vec
                (for [z (range rz) y (range ry) x (range rx)]
                  (gyroid 0.1 1.0 [x y z])))]
    (prn "voxels:" (take 100 voxels))
    voxels))

(defn make-pipeline
  [{:keys [o-buffers v-buf p-buf q-buf num] :as args}]
  (prn args)
  (ops/compile-pipeline
   :steps
   (concat
    [{:write [p-buf v-buf]}]
    (mapcat
     (fn [o-buf]
       [{:write o-buf}
        {:name "RenderImage"
         :in [v-buf o-buf]
         :out p-buf
         :n num
         :args [[num :int]]}])
     o-buffers)
    [{:write q-buf}
     {:name "TonemapImage"
      :in [p-buf (first o-buffers)]
      :out q-buf
      :n num
      :read [:out]
      :args [[num :int]]}])))

(defn make-option-buffers
  [n opts]
  (let [t-opts (sg/lookup :TRenderOptions)]
    (vec
     (for [i (range n)]
       (cl/as-clbuffer (sg/encode t-opts (render-options (assoc opts :t (* i 0.1)))) :readonly)))))

(defn init-renderer
  [& {:keys [width height vres iter] :as args}]
  (let [cl-state (cl/init-state :device :cpu :program (clu/resource-stream cl-program))]
    (cl/with-state cl-state
      (println "build log: ")
      (println :cpu (cl/build-log (:program cl-state) (cl/max-device (:ctx cl-state) :cpu)))
      ;; (println :gpu (cl/build-log (:program cl-state) (cl/max-device (:ctx cl-state) :gpu)))
      (let [num (* width height)
            state (merge
                   {:cl-state cl-state
                    :o-buffers (make-option-buffers iter args)
                    :num num}
                   (ops/init-buffers
                    1 1
                    :v-buf {:wrap (make-volume args) :type :float :usage :readonly}
                    :p-buf {:size (* num 4) :type :float :usage :readwrite}
                    :q-buf {:size num :type :int :usage :writeonly}))]
        (assoc state :pipeline (make-pipeline state))))))

(defn test-render
  [width height iter]
  (let [state (init-renderer :width width :height height :vres [128 128 128] :iter iter)]
    (cl/with-state (:cl-state state)
      (let [argb (ops/execute-pipeline (:pipeline state) :verbose true :final-size (:num state))
            img (pix/make-image width height)
            img-pix (pix/get-pixels img)]
        (prn (apply str (map #(format "%08x " %) (take 1 (cl/buffer-seq argb)))))
        (cl/rewind argb)
        (prn "copying buffer")
        (loop [cols (cl/buffer-seq argb) i 0]
          (when (< i (:num state))
            (aset-int img-pix i (first cols))
            (recur (rest cols) (inc i))))
        (pix/set-pixels img img-pix)
        (pix/save-png img "foo.png")))))