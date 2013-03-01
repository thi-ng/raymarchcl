(defproject raymarchcl "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.4.0"]
                 [com.postspectacular/structgen "0.1.2"]
                 [com.postspectacular/simplecl "0.1.6-SNAPSHOT"]
                 [com.postspectacular/piksel "0.1.4"]]
  :jvm-opts ["-server" "-Xms1g" "-Xmx2g" "-XX:+AggressiveOpts" "-XX:+UseParNewGC"])
