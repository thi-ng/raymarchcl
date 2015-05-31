(defproject thi.ng/raymarchcl "0.1.0"
  :description  "Experiment w/ OpenCL voxel raymarching via Clojure"
  :url          "http://thi.ng/raymarchcl"
  :license      {:name "Apache Software License 2.0"
                 :url "http://www.apache.org/licenses/LICENSE-2.0"
                 :distribution :repo}
  :scm          {:name "git"
                 :url "git@github.com:thi-ng/raymarchcl.git"}
  :dependencies [[org.clojure/clojure "1.7.0-RC1"]
                 [thi.ng/structgen "0.2.1"]
                 [thi.ng/simplecl "0.2.2"]
                 [thi.ng/geom "0.0.803"]
                 [thi.ng/math "0.1.0"]
                 [com.postspectacular/piksel "0.1.4"]]
  :jvm-opts ["-server" "-Xms1g" "-Xmx2g"])
