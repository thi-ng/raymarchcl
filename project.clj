(defproject raymarchcl "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0-beta3"]
                 [thi.ng/structgen "0.2.1"]
                 [thi.ng/simplecl "0.2.0"]
                 [com.postspectacular/piksel "0.1.4"]
                 [com.postspectacular/toxi2 "0.1.2-SNAPSHOT"]
                 [thi.ng/geom-core "0.0.783"]]
  :jvm-opts ["-server" "-Xms1g" "-Xmx2g" "-XX:+AggressiveOpts" "-XX:+UseParNewGC"])
