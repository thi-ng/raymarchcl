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
  [{:keys [width height vres t iter eyepos]}]
  (let [d 0.005]
    {:resolution [width height]
     :flareAmp 0.05
     :materials
     [{:albedo [1.0 1.0 1.0 1.0] :r0 0.4 :smoothness 0.1}
      {:albedo [4.9 0.9 0.01 1.0] :r0 0.1 :smoothness 0.8}
      {:albedo [1.9 1.9 1.9 1.0] :r0 0.3 :smoothness 0.4}
      {:albedo [0.01 0.01 0.01 1.0] :r0 0.1 :smoothness 0.4}]
     :maxDist 30
     :invAspect (float (/ height width))
     :eps 0.005
     :skyColor [1.8 1.8 1.8]
     :startDist 0.0
     :isoVal 0.125
     :voxelRes vres
     :maxVoxelIter 128
     :lightPos [2 3.0 2]
     :shadowBias 0.1
     :aoAmp 0.2
     :voxelBoundsMin [-0.97 -0.97 -0.97]
     :aoStepDist 0.2
     :eyePos (or eyepos [1.2 0.15 1.2])
     :aoIter 5
     :voxelSize 0.05
     :normOffsets [[d (- d) (- d) 0] [(- d) (- d) d 0] [(- d) d (- d) 0] [d d d 0]]
     :frameBlend (max 0.01 (/ 1.0 iter))
     :groundY 1.05
     :shadowIter 80
     :lightColor [56 36 16]
     :targetPos [0 0 0]
     :maxIter 128
     :dof 0.025
     :exposure 3.5
     :minLightAtt 0.0
     :voxelBounds2 [2 2 2]
     :time t
     :fogDensity 0.01
     :voxelBoundsMax [0.97 0.97 0.97]
     :lightScatter 0.2
     :invVoxelScale [0.5 0.5 0.5]
     :up [0 1 0]
     :voxelBounds [1 1 1]
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
                (for [z (range rz) y (range ry) x (range rx)
                      :let [v (gyroid 0.03 1.0 [x y z] [0.3875 0 0])]]
                  (if (< (Math/abs (- 0.2 v)) 0.05)
                    (if (< (rem x 32) 16) 0.25 0.5)
                    (if (> v 0.35) 1 0))))]
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

(defn update-option-buffers
  [buffers opts]
  (let [t-opts (sg/lookup :TRenderOptions)]
    (vec
     (for [i (range (count buffers))
           :let [b (get buffers i)]]
       (do
         (cl/rewind b)
         (cl/into-buffer b (sg/encode t-opts (render-options (assoc opts :t (* i 0.1)))))
         (cl/rewind b))))))

(defn init-renderer
  [{:keys [width height vres iter] :as args}]
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
  [width height iter res]
  (let [state (init-renderer {:width width :height height :vres [res res res] :iter iter})]
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

(defn test-anim
  [width height iter res]
  (let [args {:width width :height height :vres [res res res] :iter iter}
        img (pix/make-image width height)
        img-pix (pix/get-pixels img)
        state (init-renderer args)]
    (cl/with-state (:cl-state state)
      (time
       (doseq [frame (range 120)]
         (prn "rendering frame #" frame)
         (cl/rewind (:q-buf state))
         (let [theta (float (/ (* frame 3 Math/PI) 180.0))
               frame-args (assoc args
                            :eyepos [(* 1.5 (Math/cos theta)) 0.15 (* 1.5 (Math/sin theta))])
               _ (update-option-buffers (:o-buffers state) frame-args)
               ;;_ (prn state)
               argb (time (ops/execute-pipeline
                           (make-pipeline state)
                           :verbose false
                           :release false
                           :final-size (:num state)))]
           ;;(prn (apply str (map #(format "%08x " %) (take width (cl/buffer-seq argb)))))
           ;;(cl/rewind argb)
           (prn "copying pixel buffer")
           (loop [cols (cl/buffer-seq argb) i 0]
             (when (< i (:num state))
               (aset-int img-pix i (first cols))
               (recur (rest cols) (inc i))))
           (pix/set-pixels img img-pix)
           (pix/save-png img (format "anim-%03d.png" frame)))))
      (cl/release (:ctx (:cl-state state))))))
