(ns thi.ng.raymarchcl.meshvoxel
  (:require
   [thi.ng.raymarchcl.core :as rm]
   [thi.ng.raymarchcl.io :as vio]
   [thi.ng.geom.core :as g]
   [thi.ng.geom.core.vector :as v :refer [vec3]]
   [thi.ng.geom.core.utils :as gu]
   [thi.ng.geom.mesh.io :as mio]
   [clojure.java.io :as io]
   [piksel.core :as pix]))

(defn load-mesh
  [path]
  (-> path io/input-stream mio/read-stl))

(defn mesh-scale
  [vertices res]
  (let [[p [sx sy sz]] (gu/bounding-box vertices)
        md  (max sx sy sz)
        off (vec3 (map #(* 0.5 res (- 1.0 (/ % md))) [sx sy sz]))
        s   (vec3 (/ res md))]
    (prn :p p :s [sx sy sz] :off off :md md :scale s)
    (fn [v] (g/+ off (g/* (g/- v p) s)))))

(defn voxelize-scatter
  [vertices res]
  (let [voxels   (byte-array (* res res res))
        rxy      (* res res)
        r2       (/ res 2)
        scale-fn (mesh-scale vertices res)]
    (doseq [v vertices]
      (let [[x y z] (map int (scale-fn v))]
        (doseq [i (range (if (< (rand) 0.25) (rand 5) 1))]
          (let [dx (int (rand (* (/ i 5) r2)))
                x (int (- x dx (* res -0.4)))
                z (max (- z (int (* r2 (+ (* 0.125 (rand)) 0.125)))) 0)
                y (+ y (* res 0.4))]
            (doseq [z (range (dec z) (+ 2 z))
                    y (range (dec y) (+ 2 y))
                    x (range (dec x) (+ 2 x))]
              (when (and (>= z 0) (< z res) (>= y 0) (< y res) (>= x 0) (< x res))
                (aset-byte voxels (+ (* (int y) rxy) (* (int z) res) (int x)) 64)))))))
    voxels))

(defn voxelize-ks
  [vertices res ks]
  (let [voxels   (byte-array (* res res res))
        rxy      (* res res)
        scale-fn (mesh-scale vertices res)
        ks1      (inc ks)]
    (doseq [v vertices]
      (let [[x y z] (map int (scale-fn v))]
        ;;(prn x y z)
        (doseq [z (range (max 0 (- z ks)) (min res (+ z ks1)))
                y (range (max 0 (- y ks)) (min res (+ y ks1)))
                x (range (max 0 (- x ks)) (min res (+ x ks1)))]
          (aset-byte voxels (+ (* (int z) rxy) (* (int y) res) (int x)) -1))))
    voxels))

(defn voxelize
  [vertices res]
  (let [voxels   (byte-array (* res res res))
        rxy      (* res res)
        scale-fn (mesh-scale vertices res)]
    (doseq [v vertices]
      (let [[x y z] (map int (scale-fn v))]
        (when (and (>= z 0) (< z res) (>= y 0) (< y res) (>= x 0) (< x res))
          (aset-byte voxels (+ (* (int z) rxy) (* (int y) res) (int x)) -1))))
    voxels))

(defn make-heatmap
  [path amp]
  (let [img    (pix/load-image path)
        pixels (pix/get-pixels img)
        res    (.getWidth img)
        voxels (byte-array (* res res res))
        rxy    (* res res)]
    (doseq [y (range res) x (range res)]
      (let [c (bit-and (aget pixels (+ (* y res) x)) 255)
            h (if (pos? c) (if (> c 224) 2 (max 2 (* c amp))) 0)]
        (doseq [hh (range h)]
          (aset-byte voxels (+ (* y rxy) (* hh res) x) -1))))
    voxels))

(defn make-heatmap-anim
  [path out-path n]
  (doseq [i (range n)]
    (let [out (format out-path i)]
      (vio/save-volume out 256 (make-heatmap path (float (/ i (* n 1.33333))))))))
