(ns thi.ng.raymarchcl.generators
  (:require
   [thi.ng.geom.core :as g]
   [thi.ng.geom.core.vector :as v :refer [vec3]]
   [thi.ng.math.core :as m]
   [thi.ng.math.macros :as mm]))

(defn generate-scatter-offsets
  [num]
  (let [rnd (java.util.Random. (System/nanoTime))
        rv #(float (- (* 2.0 (.nextDouble rnd)) 1.0))]
    (->> (for [i (range num)
               :let [[x y z w :as off] (repeatedly 4 rv)
                     m (/ 1.0 (Math/sqrt (mm/madd x x y y z z w w)))]]
           (mapv #(* % m) off))
         (mapcat identity))))

(defn gyroid [s t p o]
  (let [[x y z] (g/madd p s o)]
    (- (m/abs
        (mm/madd
         (Math/cos x) (Math/sin z)
         (Math/cos y) (Math/sin x)
         (Math/cos z) (Math/sin y)))
       t)))

(defn make-gyroid-volume
  [{[rx ry rz] :vres}]
  (prn "volume " rx ry rz)
  (let [voxels (byte-array (* rx ry rz))
        rxy    (* rx ry)
        offset (vec3 0.3875 0.0 0.0)
        scl    (* 0.01 (/ 512 rx))]
    (doseq [z (range rz) y (range ry) x (range rx)]
      (when (>= (bit-and z 0x3f) 32)
        (let [v (gyroid scl 1.0 (vec3 x y z) offset)
              idx (mm/madd z rxy y rx x)]
          (when (and (zero? x) (zero? y)) (prn z))
          (if (< (m/abs (- 0.2 v)) 0.05)
            (if (< (bit-and x 0x3f) 32) (aset-byte voxels idx 64) (aset-byte voxels idx -128))
            (when (> v 0.35) (aset-byte voxels idx -1))))))
    voxels))

(defn make-terrain
  [{[rx ry rz] :vres}]
  (prn "terrain " rx ry rz)
  (let [voxels (byte-array (* rx ry rz))
        rxy (* rx ry)]
    (doseq [z (range 4) y (range (int (* ry 0.666))) x (range rx)]
      (aset-byte voxels (mm/madd z rxy y rx x) 64)
      (aset-byte voxels (mm/madd x rxy y rx (dec (- rx z))) 64))
    (doseq [z (range rz) x (range rx)]
      (let [dx (- 16 (rem x 32))
            dz (- 16 (rem z 32))
            r (+ (* dx dx) (* dz dz))]
        (when (<= r 121)
          (let [y (int (* ry (+ 0.25 (* 0.125 (* (Math/sin (* z 0.02)) (Math/cos (* x 0.03)))))))]
            (doseq [yy (range (inc y))]
              (aset-byte voxels (mm/madd z rxy yy rx x) -1))))))
    voxels))
