(ns thi.ng.raymarchcl.materials)

(def presets
  {:orange-stripes
   {:lightColor  [[28 18 8 0] [8 18 28 0]]
    :lightPos    [[-2 0 -2 0] [2 0 2 0]]
    :materials   [{:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.1
                   :smoothness 0.9}
                  {:albedo [4.9 0.9 0.05 1.0]
                   :r0 0.01
                   :smoothness 0.5}
                  {:albedo [1.9 1.9 1.9 1.0]
                   :r0 0.01
                   :smoothness 0.4}
                  {:albedo [0.9 0.9 0.9 1.0]
                   :r0 0.8
                   :smoothness 0.1}]
    :numLights   2
    :aoAmp       0.25
    :reflectIter 1}

   :metal
   {:lightColor  [[28 18 8 0] [16 36 56 0]]
    :lightPos    [[0 2 0 0] [3 0 3 0]]
    :materials   [{:albedo [0.01 0.01 0.01 1.0]
                   :r0 0.1
                   :smoothness 0.5}
                  {:albedo [1.9 1.9 1.9 1.0]
                   :r0 0.1
                   :smoothness 0.5}
                  {:albedo [0.25 0.27 0.5 1.0]
                   :r0 0.7
                   :smoothness 0.1}
                  {:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.2
                   :smoothness 0.1}]
    :numLights   2
    :aoAmp       0.25
    :reflectIter 3}

   :metal2
   {:lightColor  [[28 18 8 0] [8 18 28 0]]
    :lightPos    [[-2 0 -2 0] [2 0 2 0]]
    :materials   [{:albedo [0.0 0.0 0.0 1.0]
                   :r0 0.1
                   :smoothness 0.9}
                  {:albedo [1.0 1.01 1.075 1.0]
                   :r0 0.4
                   :smoothness 0.7}
                  {:albedo [1.9 1.9 1.9 1.0]
                   :r0 0.4 :smoothness 0.5}
                  {:albedo [0.9 0.9 0.9 1.0]
                   :r0 0.75
                   :smoothness 0.2}]
    :numLights   2
    :aoAmp       0.25
    :reflectIter 3}
   
   :ao
   {:lightColor  [[50 50 50 0]]
    :materials   [{:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.0
                   :smoothness 1.0}
                  {:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.0
                   :smoothness 1.0}
                  {:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.0
                   :smoothness 1.0}
                  {:albedo [1.0 1.0 1.0 1.0]
                   :r0 0.0
                   :smoothness 1.0}]
    :numLights   1
    :aoAmp       0.25
    :reflectIter 0}})
