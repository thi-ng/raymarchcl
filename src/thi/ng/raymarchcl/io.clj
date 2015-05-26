(ns thi.ng.raymarchcl.io
  (:require
   [thi.ng.simplecl.core :as cl]
   [thi.ng.simplecl.ops :as ops]
   [clojure.java.io :as io])
  (:import
   [java.nio ByteBuffer]))

(defn save-volume
  [path res ^bytes voxels]
  (with-open [out (java.io.DataOutputStream. (io/output-stream path))]
    (.write out (byte-array (map byte "VOXEL")) 0 5) ;; magic: VOXEL
    (.writeInt out res)                              ;; resx
    (.writeInt out res)                              ;; resy
    (.writeInt out res)                              ;; resz
    (.writeByte out 1)                               ;; element size in bytes
    (.write out voxels 0 (count voxels))))

(defn load-volume
  [path]
  (with-open [in (java.io.DataInputStream. (io/input-stream path))]
    (.read in (byte-array 5) 0 5) ; magic: VOXEL
    (let [x   (.readInt in) ; resx
          y   (.readInt in) ; resy
          z   (.readInt in) ; resz
          s   (.readByte in)
          vox (byte-array (* x y z))]
      (.read in vox 0 (count vox))
      (->> (ops/init-buffers
            1 1
            :v-buf {:wrap (ByteBuffer/wrap vox) :usage :readonly})
           :v-buf
           (cl/rewind)))))
