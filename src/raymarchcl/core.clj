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
  [{:keys [width height vres t iter]}]
  (let [d 0.005]
    {:resolution [width height],
     :flareAmp 0.05,
     :materials
     [{:albedo [1.0 1.0 1.0 1.0], :r0 0.4, :smoothness 0.1}
      {:albedo [4.9 0.5 0.05 1.0], :r0 0.2, :smoothness 0.8}],
     :maxDist 30,
     :invAspect (float (/ height width)),
     :eps 0.005,
     :skyColor [1.8 1.8 1.8],
     :startDist 0.0,
     :isoVal 0.125,
     :voxelRes vres
     :maxVoxelIter 96,
     :lightPos [2 3.0 -2],
     :shadowBias 0.1,
     :aoAmp 0.2,
     :voxelBoundsMin [-0.97 -0.97 -0.97],
     :aoStepDist 0.2,
     :eyePos [-2.25 0.375 2.25],
     :aoIter 5,
     :voxelSize 0.05,
     :normOffsets [[d (- d) (- d) 0] [(- d) (- d) d 0] [(- d) d (- d) 0] [d d d 0]],
     :frameBlend (max 0.01 (/ 1.0 iter)),
     :groundY 1.05,
     :shadowIter 80,
     :lightColor [56 36 16],
     :targetPos [0 0 0],
     :maxIter 64,
     :dof 0.1,
     :exposure 3.5,
     :minLightAtt 0.0,
     :voxelBounds2 [2 2 2],
     :time t,
     :fogDensity 0.01,
     :voxelBoundsMax [0.97 0.97 0.97],
     :lightScatter 0.2,
     :invVoxelScale [0.5 0.5 0.5],
     :up [0 1 0],
     :voxelBounds [1 1 1],
     :gamma 1.5}))

(defn gyroid [s t p o]
  "Evaluates gyroid function at scaled point `v`."
  (let [[x y z] (map #(+ (* s %) %2) p o)]
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
                  (if (< (Math/abs (- 0.2 (gyroid 0.0165 1.0 [x y z] [0.3875 -0.6 0.2]))) 0.1)
                    1 0)))]
    ;;(prn "voxels:" (take 100 voxels))
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
            state (time
                   (merge
                    {:cl-state cl-state
                     :o-buffers (make-option-buffers iter args)
                     :num num}
                    (ops/init-buffers
                     1 1
                     :v-buf {:wrap (make-volume args) :type :float :usage :readonly}
                     :p-buf {:size (* num 4) :type :float :usage :readwrite}
                     :q-buf {:size num :type :int :usage :writeonly})))]
        (assoc state :pipeline (make-pipeline state))))))

(defn test-render
  [width height iter]
  (let [state (init-renderer :width width :height height :vres [224 224 224] :iter iter)]
    (cl/with-state (:cl-state state)
      (let [argb (time (ops/execute-pipeline (:pipeline state) :verbose false :final-size (:num state)))
            img (pix/make-image width height)
            img-pix (pix/get-pixels img)]
        ;;(prn (apply str (map #(format "%08x " %) (take width (cl/buffer-seq argb)))))
        ;;(cl/rewind argb)
        (prn "copying buffer")
        (loop [cols (cl/buffer-seq argb) i 0]
          (when (< i (:num state))
            (aset-int img-pix i (first cols))
            (recur (rest cols) (inc i))))
        (pix/set-pixels img img-pix)
        (pix/save-png img "foo.png")))))