(ns raymarchcl.core
  (:require
   [simplecl.core :as cl]
   [simplecl.utils :as clu]
   [simplecl.ops :as ops]
   [structgen.core :as sg]
   [structgen.parser :as sp]
   [piksel.core :as pix]
   [clojure.java.io :as io]
   [toxi.math.core :as m])
  (:import
   [java.nio ByteBuffer IntBuffer]))

(def cl-program
  "Location of the OpenCL program"
  "renderer.cl")

;; parse typedefs in OpenCL program and register all found struct types
(sg/reset-registry!)
(sg/register! (sp/parse-specs (slurp (clu/resource-stream cl-program))))

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

(defn montecarlo
  [num]
  (let [rnd (java.util.Random. (System/nanoTime))
        rv #(float (- (* 2.0 (.nextDouble rnd)) 1.0))]
    (flatten
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
    :lightPos [[-2 0 -2 0] [2 0 2 0]]
    :numLights 2
    :materials [{:albedo [1.0 1.0 1.0 1.0] :r0 0.4 :smoothness 0.9}
                {:albedo [4.9 0.9 0.05 1.0] :r0 0.1 :smoothness 0.8}
                {:albedo [1.9 1.9 1.9 1.0] :r0 0.3 :smoothness 0.4}
                {:albedo [0.9 0.9 0.9 1.0] :r0 0.8 :smoothness 0.1}]
    :aoAmp 0.1875
    :reflectIter 3}
   :metal
   {:lightColor [[28 18 8 0] [16 36 56 0]]
    :lightPos [[0 2 0 0] [3 0 3 0]]
    :numLights 2
    :materials [{:albedo [0.01 0.01 0.01 1.0] :r0 0.1 :smoothness 0.5}
                {:albedo [1.9 1.9 1.9 1.0] :r0 0.1 :smoothness 0.5}
                {:albedo [0.9 0.9 0.9 1.0] :r0 0.75 :smoothness 0.2}
                {:albedo [1.0 1.0 1.0 1.0] :r0 0.2 :smoothness 0.1}]
    :aoAmp 0.25
    :reflectIter 3}
   :metal2
   {:lightColor [[28 18 8 0] [8 18 28 0]]
    :lightPos [[-2 0 2 0] [2 0 2 0]]
    :numLights 2
    :materials [{:albedo [0.0 0.0 0.0 1.0] :r0 0.1 :smoothness 0.9}
                {:albedo [1.0 1.01 1.075 1.0] :r0 0.4 :smoothness 0.7}
                {:albedo [1.9 1.9 1.9 1.0] :r0 0.4 :smoothness 0.5}
                {:albedo [0.9 0.9 0.9 1.0] :r0 0.75 :smoothness 0.2}]
    :aoAmp 0.25
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
  [{:keys [width height vres t iter eyepos mat fov targetpos]}]
  (let [eps 0.005
        d (* eps 1.1)
        clip 0.99]
    (merge
     {:resolution [width height]
      :fov (or fov 2)
      :flareAmp 0.015
      :maxDist 30
      :invAspect (float (/ height width))
      :eps eps
      :skyColor1 [1.8 1.8 1.9]
      :skyColor2 [0.1 0.1 0.1]
      :startDist 0.0
      :isoVal 32
      :voxelRes (conj vres (* (vres 0) (vres 1)))
      :maxVoxelIter 192
      :numLights 2
      :lightPos [[-2 0 -2 0] [2 0 2 0]]
      :shadowBias 0.1
      :aoAmp 0.2
      :aoMaxAmp 1 ;;0.1875
      :voxelBoundsMin [(- clip) (- clip) (- clip)]
      :aoStepDist 0.05
      :eyePos (or eyepos [2 0 2])
      :aoIter 5
      :voxelSize 0.025
      :normOffsets [[d (- d) (- d) 0] [(- d) (- d) d 0] [(- d) d (- d) 0] [d d d 0]]
      :frameBlend (/ 1.0 iter)
      :groundY 1.001
      :shadowIter 128
      :lightColor [50 50 50]
      :targetPos (or targetpos [0 -0.15 0])
      :maxIter 128
      :reflectIter 0
      :dof 0.005
      :exposure 3.5
      :minLightAtt 0.0
      :voxelBounds2 [2 2 2]
      :time t
      :fogPow 0.05
      :voxelBoundsMax [clip clip clip]
      :lightScatter 0.2
      :invVoxelScale [0.5 0.5 0.5]
      :up [0 1 0]
      :voxelBounds [1 1 1]
      :gamma 1.5}
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
            (if (< (bit-and x 0x3f) 32) (aset-byte voxels idx 64) (aset-byte voxels idx 0))
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
  [{:keys [o-buffers v-buf p-buf q-buf mc-buffers num] :as args}]
  (ops/compile-pipeline
   :steps
   (concat
    [{:write [p-buf v-buf]}]
    (mapcat
     (fn [o-buf mc-buf]
       [{:write [o-buf mc-buf]}
        {:name "RenderImage"
         :in [v-buf mc-buf o-buf]
         :out p-buf
         :n num
         :args [[num :int]]}])
     o-buffers mc-buffers)
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
  [{:keys [width height vres iter vname] :as args}]
  (let [pl (first (cl/available-platforms))
        dev (cl/max-device pl)
        ctx (cl/make-context [dev])
        cl-state (cl/init-state
                  :platform pl
                  :context ctx
                  :device dev
                  :program [(clu/resource-stream cl-program) :fast-math :enable-mad])]
    (cl/with-state cl-state
      (println "build log:\n-------------------")
      (println (cl/build-log))
      (let [num (* width height)
            state (time
                   (merge
                    {:cl-state cl-state
                     :o-buffers (make-option-buffers iter args)
                     :mc-buffers (for [i (range iter)]
                                   (cl/as-clbuffer :float (montecarlo 0x4000) :readonly))
                     :num num}
                    (ops/init-buffers
                     1 1
                     ;;:v-buf {:wrap (make-volume args) :type :byte :usage :readonly}
                     ;;:v-buf {:wrap (make-terrain args) :type :byte :usage :readonly}
                     :p-buf {:size (* num 4) :type :float :usage :readwrite}
                     :q-buf {:size num :type :int :usage :writeonly}
                     ;;:mc-buf {:wrap (montecarlo 0x4000) :type :float
                                        ;:usage :readonly}
                     )
                    ;;{:v-buf (load-volume "gyroid-512.vox")}
                    {:v-buf (load-volume (or vname "gyroid-sliced-512-s0.01.vox"))}
                    ;;{:v-buf (load-volume "terrain-512-solid.vox")}
                    ))]
        (assoc state :pipeline (make-pipeline state))))))

(defn test-render
  [& {:keys [width height iter vres mat vname out-path theta dist]
      :or {width 640 height 360 iter 1 vres 256 mat :metal out-path "foo.png"
           theta 135 dist 2.25}}]
  (let [state (init-renderer {:width width :height height
                              :vres [vres vres vres]
                              :iter iter
                              ;;:eyepos (compute-eyepos (* 1.7 45) 0.66 0.5)
                              :eyepos (compute-eyepos theta dist 0.35)
                              :targetpos [0.25 -0.4 0]
                              :mat mat
                              :vname vname})]
    (cl/with-state (:cl-state state)
      (let [argb (time (ops/execute-pipeline (:pipeline state) :verbose false :final-size (:num state)))
            img (pix/make-image width height)
            img-pix (pix/get-pixels img)]
        (prn "copying buffer")
        (.get ^IntBuffer argb img-pix)
        (comment
          (loop [cols (cl/buffer-seq argb) i 0]
            (when (< i (:num state))
              (aset-int img-pix i (first cols))
              (recur (rest cols) (inc i)))))
        (pix/set-pixels img img-pix)
        (pix/save-png img out-path)))))

(defn test-anim
  [width height iter res mat & vname]
  (let [args {:width width :height height :vres [res res res]
              :iter iter
              :mat mat
              :vname (first vname)}
        img (pix/make-image width height)
        img-pix (pix/get-pixels img)
        state (init-renderer args)]
    (cl/with-state (:cl-state state)
      (time
       (doseq [frame (range 13 14)]
         (prn "rendering frame #" frame)
         (cl/rewind (:q-buf state))
         (let [t (m/map-interval frame 0 35 0.0 1.0)
               theta (m/map-interval t 0 1 0 350)
               r (m/map-interval t 0 1 2.25 2.25)
               y (m/map-interval t 0 1 0.44 0.45)
               ty (m/map-interval t 0 1 -0.15 -0.15)
               fov (m/map-interval t 0 1 2 2)
               frame-args (assoc args :fov fov :targetpos [0 ty 0] :eyepos (compute-eyepos theta r y))
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
