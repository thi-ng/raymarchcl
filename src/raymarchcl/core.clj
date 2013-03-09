(ns raymarchcl.core
  (:require
   [simplecl.core :as cl]
   [simplecl.utils :as clu]
   [simplecl.ops :as ops]
   [structgen.core :as sg]
   [structgen.parser :as sp]
   [piksel.core :as pix]
   [clojure.java.io :as io])
  (:import [java.nio ByteBuffer]))

(def cl-program
  "Location of the OpenCL program"
  "renderer.cl")

;; parse typedefs in OpenCL program and register all found struct types
(sg/reset-registry!)
(sg/register! (sp/parse-specs (slurp (clu/resource-stream cl-program))))

(defn montecarlo
  [num]
  (let [rnd (java.util.Random. (System/nanoTime))
        rv #(float (- (* 2.0 (.nextDouble rnd)) 1.0))]
    (vec
     (for [i (range num)
           :let [x (rv) y (rv) z (rv) w (rv)
                 m (/ 1.0 (Math/sqrt (+ (* x x) (* y y) (* z z) (* w w))))]]
       [(* x m) (* y m) (* z m) (* w m)]))))

(defn compute-eyepos
  [theta rad y]
  (let [theta (Math/toRadians theta)]
    [(* rad (Math/cos theta)) y (* rad (Math/sin theta))]))

(def material-presets
  {:orange-stripes
   {:lightColor [[28 18 8 0] [8 18 28 0]]
    :numLights 2
    :materials [{:albedo [1.0 1.0 1.0 1.0] :r0 0.4 :smoothness 0.9}
                {:albedo [4.9 0.9 0.05 1.0] :r0 0.1 :smoothness 0.8}
                {:albedo [1.9 1.9 1.9 1.0] :r0 0.3 :smoothness 0.4}
                {:albedo [0.9 0.9 0.9 1.0] :r0 0.8 :smoothness 0.1}]
    :aoAmp 0.1875
    :reflectIter 3}
   :metal
   {:lightColor [[28 18 8 0]]
    :numLights 1
    :materials [{:albedo [1.0 1.0 1.0 1.0] :r0 0.4 :smoothness 0.9}
                {:albedo [0.0 0.01 0.05 1.0] :r0 0.2 :smoothness 0.7}
                {:albedo [1.9 1.9 1.9 1.0] :r0 0.3 :smoothness 0.4}
                {:albedo [0.9 0.9 0.9 1.0] :r0 0.75 :smoothness 0.2}]
    :aoAmp 0.1875
    :reflectIter 3}
   :ao
   {:lightColor [[50 50 50 0]]
    :numLights 1
    :materials [{:albedo [1.0 1.0 1.0 1.0] :r0 0.0 :smoothness 1.0}
                {:albedo [1.0 1.0 1.0 1.0] :r0 0.0 :smoothness 1.0}
                {:albedo [1.0 1.0 1.0 1.0] :r0 0.0 :smoothness 1.0}
                {:albedo [1.0 1.0 1.0 1.0] :r0 0.0 :smoothness 1.0}]
    :aoAmp 0.25
    :reflectIter 0}})

(defn render-options
  [{:keys [width height vres t iter eyepos mat]}]
  (let [eps 0.005
        d (* eps 1.05)
        clip 0.99]
    (merge
     {:resolution [width height]
      :fov 2
      :flareAmp 0.015
      :maxDist 30
      :invAspect (float (/ height width))
      :eps eps
      :skyColor1 [0.5 1.5 3.0]
      :skyColor2 [1.8 1.8 1.8]
      :startDist 0.0
      :isoVal 32
      :voxelRes vres
      :maxVoxelIter 192
      :lightPos [[-2 0 -2 0] [2 0 2 0]]
      :numLights 2
      :shadowBias 0.1
      :aoAmp 0.1
      :aoMaxAmp 0.1875
      :voxelBoundsMin [(- clip) (- clip) (- clip)]
      :aoStepDist 0.05
      :eyePos (or eyepos [2 0 2])
      :aoIter 20
      :voxelSize 0.05
      :normOffsets [[d (- d) (- d) 0] [(- d) (- d) d 0] [(- d) d (- d) 0] [d d d 0]]
      :frameBlend (/ 1.0 iter)
      :groundY 1.1
      :shadowIter 32
      :lightColor [50 50 50]
      :targetPos [0 -0.124 0]
      :maxIter 80
      :reflectIter 0
      :dof 0.01
      :exposure 3.5
      :minLightAtt 0.0
      :voxelBounds2 [2 2 2]
      :time t
      :fogDensity 0.01
      :voxelBoundsMax [clip clip clip]
      :lightScatter 0.2
      :invVoxelScale [0.5 0.5 0.5]
      :up [0 1 0]
      :voxelBounds [1 1 1]
      :gamma 1.5
      :mcSamples (montecarlo 0x4000)}
     (get material-presets mat (material-presets :ao)))))

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
  (let [voxels (byte-array (* rx ry rz))
        rxy (* rx ry)]
    (doseq [z (range rz) y (range ry) x (range rx)]
      (when (>= (bit-and z 0x3f) 32)
        (let [v (gyroid 0.01 1.0 [x y z] [0.3875 0.0 0.0])
	      idx (+ (+ (* z rxy) (* y rx)) x)]
          (when (and (zero? x) (zero? y)) (prn z))
            (if (< (Math/abs (- 0.2 v)) 0.05)
              (if (< (bit-and x 0x3f) 32) (aset-byte voxels idx 64) (aset-byte voxels idx 127))
              (when (> v 0.35) (aset-byte voxels idx -1))))))
    voxels))

(defn make-terrain
  [{[rx ry rz] :vres}]
  (prn "terrain " rx ry rz)
  (let [voxels (byte-array (* rx ry rz))
        rxy (* rx ry)]
    (doseq [z (range 4) y (range (int (* ry 0.666))) x (range rx)]
      (aset-byte voxels (+ (* z rxy) (* y rx) x) 64)
      (aset-byte voxels (+ (* x rxy) (+ (* y rx) (- (- rx z) 1))) 64))
    (doseq [z (range rz) x (range rx)]
      (let [dx (- 16 (rem x 32))
            dz (- 16 (rem z 32))
            r (+ (* dx dx) (* dz dz))]
        (when (<= r 121)
          (let [y (int (* ry (+ 0.25 (* 0.125 (* (Math/sin (* z 0.02)) (Math/cos (* x 0.03)))))))]
            (doseq [yy (range (inc y))]
              (aset-byte voxels (+ (* z rxy) (+ (* yy rx) x)) -1))))))
    voxels))

(defn save-volume
  [path res ^bytes voxels]
  (with-open [out (java.io.DataOutputStream. (io/output-stream path))]
    (.write out (byte-array (map byte [86 79 88 69 76])) 0 5) ; magic: VOXEL
    (.writeInt out res) ; resx
    (.writeInt out res) ; resy
    (.writeInt out res) ; resz
    (.writeByte out 1)     ; element size in bytes
    (.write out voxels 0 (count voxels))))

(defn load-volume
  [path]
  (with-open [in (java.io.DataInputStream. (io/input-stream path))]
    (.read in (byte-array 5) 0 5) ; magic: VOXEL
    (let [x (.readInt in) ; resx
          y (.readInt in) ; resy
          z (.readInt in) ; resz
          s (.readByte in)
          vox (byte-array (* x y z))
          _ (.read in vox 0 (count vox))
          {:keys [v-buf]} (ops/init-buffers
                           1 1
                           :v-buf {:wrap (ByteBuffer/wrap vox) :usage :readonly})]
      (cl/rewind v-buf))))

(defn make-pipeline
  [{:keys [o-buffers v-buf p-buf q-buf num] :as args}]
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
       (cl/as-clbuffer
        (sg/encode t-opts (render-options (assoc opts :t (* i 0.333))))
        :readonly)))))

(defn update-option-buffers
  [buffers opts]
  (let [t-opts (sg/lookup :TRenderOptions)]
    (vec
     (for [i (range (count buffers))
           :let [b (get buffers i)]]
       (do
         (cl/rewind b)
         (cl/into-buffer b (sg/encode t-opts (render-options (assoc opts :t (* i 0.3333)))))
         (cl/rewind b))))))

(defn init-renderer
  [{:keys [width height vres iter] :as args}]
  (let [cl-state (cl/init-state :device :cpu :program (clu/resource-stream cl-program))]
    (cl/with-state cl-state
      (println "build log: ")
      (println :cpu (cl/build-log (:program cl-state) (cl/max-device (:ctx cl-state) :cpu)))
      ;;(println :gpu (cl/build-log (:program cl-state) (cl/max-device (:ctx cl-state) :gpu)))
      (let [num (* width height)
            state (time
                   (merge
                    {:cl-state cl-state
                     :o-buffers (make-option-buffers iter args)
                     :num num}
                    (ops/init-buffers
                     1 1
                     ;;:v-buf {:wrap (make-volume args) :type :byte :usage :readonly}
                     ;;:v-buf {:wrap (make-terrain args) :type :byte :usage :readonly}
                     :p-buf {:size (* num 4) :type :float :usage :readwrite}
                     :q-buf {:size num :type :int :usage :writeonly})
                    ;;{:v-buf (load-volume "gyroid-512.vox")}
                    ;;{:v-buf (load-volume "gyroid-sliced-256-s0.02.vox")}
                    {:v-buf (load-volume "gyroid-sliced-512-s0.01.vox")}
                    ;;{:v-buf (load-volume "terrain-512-solid.vox")}
                    ))]
        (assoc state :pipeline (make-pipeline state))))))

(defn test-render
  [width height iter res mat]
  (let [state (init-renderer {:width width :height height
                              :vres [res res res]
                              :iter iter
                              :eyepos (compute-eyepos (* 1.7 45) 0.66 0.5)
                              ;;:eyepos (compute-eyepos (* 3 45) 2.5 1)
                              :mat mat})]
    (cl/with-state (:cl-state state)
      (let [argb (time (ops/execute-pipeline (:pipeline state) :verbose false :final-size (:num state)))
            img (pix/make-image width height)
            img-pix (pix/get-pixels img)]
        (prn "copying buffer")
        (loop [cols (cl/buffer-seq argb) i 0]
          (when (< i (:num state))
            (aset-int img-pix i (first cols))
            (recur (rest cols) (inc i))))
        (pix/set-pixels img img-pix)
        (pix/save-png img "foo.png")))))

(defn test-anim
  [width height iter res mat]
  (let [args {:width width :height height :vres [res res res] :iter iter :mat mat}
        img (pix/make-image width height)
        img-pix (pix/get-pixels img)
        state (init-renderer args)]
    (cl/with-state (:cl-state state)
      (time
       (doseq [frame (range 32)]
         (prn "rendering frame #" frame)
         (cl/rewind (:q-buf state))
         (let [frame-args (assoc args :eyepos (compute-eyepos (* frame 12.125) 2.25 0.25))
               _ (update-option-buffers (:o-buffers state) frame-args)
               argb (time (ops/execute-pipeline
                           (make-pipeline state)
                           :verbose false
                           :release false
                           :final-size (:num state)))]
           (prn "copying pixel buffer")
           (loop [cols (cl/buffer-seq argb) i 0]
             (when (< i (:num state))
               (aset-int img-pix i (first cols))
               (recur (rest cols) (inc i))))
           (pix/set-pixels img img-pix)
           (pix/save-png img (format "anim/foo-%03d.png" frame)))))
      (cl/release (:ctx (:cl-state state))))))

(defn trilin
  [v000 v001 v010 v011 v100 v101 v110 v111 fx fy fz]
  (let [mix #(+ % (* (- %2 %) %3))
        vx (mix v000 v100 fx)
        vy (mix v001 v101 fx)
        vz (mix v010 v110 fx)
        vw (mix v011 v111 fx)
        i1 (mix vx vz fy)
        i2 (mix vy vw fy)]
    (mix i1 i2 fz)))
