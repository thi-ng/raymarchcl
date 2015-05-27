(ns thi.ng.raymarchcl.core
  (:require
   [thi.ng.raymarchcl.materials :as materials]
   [thi.ng.raymarchcl.generators :as gen]
   [thi.ng.raymarchcl.io :as vio]
   [thi.ng.simplecl.core :as cl]
   [thi.ng.simplecl.utils :as clu]
   [thi.ng.simplecl.ops :as ops]
   [thi.ng.structgen.core :as sg]
   [thi.ng.structgen.parser :as sp]
   [thi.ng.geom.core :as g]
   [thi.ng.geom.core.vector :as v :refer [vec3]]
   [thi.ng.math.core :as m]
   [thi.ng.math.macros :as mm]
   [piksel.core :as pix]
   [clojure.java.io :as io])
  (:import
   [java.nio ByteBuffer IntBuffer]))

(def cl-program
  "Location of the OpenCL program"
  "renderer.cl")

;; parse typedefs in OpenCL program and register all found struct types
(sg/reset-registry!)
(sg/register! (sp/parse-specs (slurp (clu/resource-stream cl-program))))

(defn render-options
  [{:keys [width height vres t iter eyepos mat fov dof targetpos gamma groundY voxelSize] :as opts}]
  (let [eps  0.005
        clip 0.99
        vres (if (number? vres) [vres vres vres] vres)]
    (merge
     {:aoAmp 0.2
      :aoIter 5
      :aoStepDist 0.05
      :dof (or dof 0.001)
      :eps eps
      :exposure 3.5
      :eyePos (or eyepos [2 0 2])
      :flareAmp 0.015
      :fogPow 0.05
      :fov (m/radians (or fov 90))
      :frameBlend (/ 1.0 iter)
      :gamma (or gamma 1.5)
      :groundY (or groundY 1.05)
      :invAspect (float (/ height width))
      :invVoxelScale [0.5 0.5 0.5]
      :isoVal 32
      :lightColor [50 50 50]
      :lightPos [[-2 0 -2 0] [2 0 2 0]]
      :lightScatter 0.2
      :maxDist 30
      :maxIter 128
      :maxVoxelIter 192
      :minLightAtt 0.0
      :numLights 2
      :reflectIter 0
      :resolution [width height]
      :shadowBias 0.1
      :shadowIter 128
      :skyColor1 [1.8 1.8 1.9]
      :skyColor2 [0.1 0.1 0.1]
      :startDist 0.0
      :targetPos (or targetpos [0 -0.15 0])
      :time t
      :up [0 1 0]
      :voxelBounds [1 1 1]
      :voxelBounds2 [2 2 2]
      :voxelBoundsMax [clip clip clip]
      :voxelBoundsMin [(- clip) (- clip) (- clip)]
      :voxelRes (conj vres (* (vres 0) (vres 1)))
      :voxelSize (or voxelSize (/ (first vres)))}
     (get materials/presets mat (materials/presets :ao)))))

(defn make-pipeline
  [{:keys [opts-buffers v-buf p-buf q-buf mc-buffers num] :as args}]
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
     opts-buffers mc-buffers)
    [{:write q-buf}
     {:name "TonemapImage"
      :in [p-buf (first opts-buffers)]
      :out q-buf
      :n num
      :read [:out]
      :args [[num :int]]}])))

(defn make-render-option-buffer
  [n opts]
  (let [t-opts (sg/lookup :TRenderOpts)]
    (vec
     (for [i (range n)]
       (cl/as-clbuffer
        (sg/encode t-opts (render-options (assoc opts :t (* i 0.333))))
        :readonly)))))

(defn update-render-option-buffer
  [buffers opts]
  (let [t-opts (sg/lookup :TRenderOpts)]
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
                    {:cl-state     cl-state
                     :opts-buffers (make-render-option-buffer iter args)
                     :mc-buffers   (for [i (range iter)]
                                     (cl/as-clbuffer :float (gen/generate-scatter-offsets 0x4000) :readonly))
                     :num          num}
                    (ops/init-buffers
                     1 1
                     ;;:v-buf {:wrap (gen/make-gyroid-volume args) :type :byte :usage :readonly}
                     ;;:v-buf {:wrap (make-terrain args) :type :byte :usage :readonly}
                     :p-buf {:size (* num 4) :type :float :usage :readwrite}
                     :q-buf {:size num :type :int :usage :writeonly})
                    {:v-buf (vio/load-volume (or vname "gyroid-sliced-512-s0.01.vox"))}
                    ))]
        (assoc state :pipeline (make-pipeline state))))))

(defn compute-eyepos
  [theta dist y]
  (g/rotate-y (vec3 0 y dist) (m/radians theta)))

(defn test-render
  [& {:keys [width height iter vres mat vname out-path theta dist]
      :or {width 640 height 360 iter 1 vres 256 mat :metal out-path "foo.png"
           theta 135 dist 2.25}
      :as opts}]
  (let [state (init-renderer
               (merge
                {:width width
                 :height height
                 :vres vres
                 :iter iter
                 :eyepos (compute-eyepos theta dist 0.35)
                 :targetpos [0 -0.4 0]
                 :mat mat
                 :vname vname}
                opts))]
    (cl/with-state (:cl-state state)
      (let [argb (time (ops/execute-pipeline (:pipeline state) :verbose false :final-size (:num state)))
            img (pix/make-image width height)
            img-pix (pix/get-pixels img)]
        (prn "copying buffer")
        (time
         (do
           (.get ^IntBuffer argb img-pix)
           (pix/set-pixels img img-pix)
           (pix/save-png img out-path)))))))

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
       (doseq [frame (range 0 35)]
         (prn "rendering frame #" frame)
         (cl/rewind (:q-buf state))
         (let [t (m/map-interval frame 0 35 0.0 1.0)
               theta (m/map-interval t 0 1 0 350)
               r (m/map-interval t 0 1 2.25 2.25)
               y (m/map-interval t 0 1 0.44 0.45)
               ty (m/map-interval t 0 1 -0.15 -0.15)
               fov (m/map-interval t 0 1 115 115)
               frame-args (assoc args :fov fov :targetpos [0 ty 0] :eyepos (compute-eyepos theta r y))
               _ (update-render-option-buffer (:opts-buffers state) frame-args)
               argb (time
                     (ops/execute-pipeline
                      (make-pipeline state)
                      :verbose false
                      :release false
                      :final-size (:num state)))]
           (prn "copying pixel buffer")
           (.get ^IntBuffer argb img-pix)
           (pix/set-pixels img img-pix)
           (pix/save-png img (format "export/frame-%04d.png" frame)))))
      (cl/release (:ctx (:cl-state state))))))
