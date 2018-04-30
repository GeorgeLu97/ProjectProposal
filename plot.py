import matplotlib.pyplot as plt
import numpy as np

average = [-0.006, 0.072, 0.14, 0.042, 0.084, 0.112, 0.124, 0.216, 0.144, 0.18, 0.14, 0.116, 0.186, 0.094, 0.088, 0.096, 0.15, 0.108, 0.132, 0.16, 0.166, 0.17, 0.226, 0.182, 0.168, 0.114, 0.126, 0.14, 0.196, 0.19, 0.134, 0.164, 0.25, 0.166, 0.112, 0.168, 0.172, 0.16, 0.188, 0.114, 0.226, 0.262, 0.168, 0.162, 0.166, 0.194, 0.174, 0.25, 0.256, 0.256, 0.188, 0.264, 0.248, 0.28, 0.258, 0.204, 0.256, 0.232, 0.246, 0.274, 0.34, 0.186, 0.246, 0.27, 0.27, 0.234, 0.32, 0.35, 0.256, 0.356, 0.312, 0.276]

townie = [0.25738396624472576, 0.25, 0.308300395256917, 0.18181818181818182, 0.2701612903225806, 0.28957528957528955, 0.2627118644067797, 0.383399209486166, 0.31153846153846154, 0.2603305785123967, 0.23333333333333334, 0.32894736842105265, 0.32388663967611336, 0.24096385542168675, 0.20532319391634982, 0.3076923076923077, 0.2972972972972973, 0.22040816326530613, 0.273542600896861, 0.30980392156862746, 0.33067729083665337, 0.308300395256917, 0.35384615384615387, 0.2923076923076923, 0.3632286995515695, 0.2804878048780488, 0.1821705426356589, 0.2831858407079646, 0.34146341463414637, 0.3045267489711934, 0.228, 0.25, 0.36470588235294116, 0.35443037974683544, 0.2962962962962963, 0.2748091603053435, 0.3244274809160305, 0.2641509433962264, 0.34156378600823045, 0.2938775510204082, 0.3345323741007194, 0.4524714828897338, 0.34509803921568627, 0.2701612903225806, 0.29554655870445345, 0.3545816733067729, 0.276, 0.4024390243902439, 0.4672489082969432, 0.35080645161290325, 0.32421875, 0.3968253968253968, 0.39148936170212767, 0.40160642570281124, 0.35365853658536583, 0.3686440677966102, 0.31906614785992216, 0.33613445378151263, 0.38783269961977185, 0.468, 0.4103585657370518, 0.2384937238493724, 0.3038461538461538, 0.40086206896551724, 0.42168674698795183, 0.42578125, 0.49586776859504134, 0.46747967479674796, 0.34274193548387094, 0.4523809523809524, 0.44635193133047213, 0.452991452991453]

mafia = [-0.24334600760456274, -0.10887096774193548, -0.032388663967611336, -0.10121457489878542, -0.0992063492063492, -0.07883817427385892, 0.0, 0.044534412955465584, -0.0375, 0.10465116279069768, 0.05384615384615385, -0.0625, 0.05138339920948617, -0.05179282868525897, -0.04219409282700422, -0.09022556390977443, -0.008298755186721992, 0.0, 0.018050541516245487, 0.004081632653061225, 0.0, 0.02834008097165992, 0.0875, 0.0625, 0.010830324909747292, -0.047244094488188976, 0.06611570247933884, 0.021897810218978103, 0.05511811023622047, 0.08171206225680934, 0.04, 0.05092592592592592, 0.1306122448979592, -0.0038022813688212928, -0.0622568093385214, 0.05042016806722689, 0.004201680672268907, 0.0425531914893617, 0.042801556420233464, -0.058823529411764705, 0.09009009009009009, 0.05063291139240506, -0.0163265306122449, 0.05555555555555555, 0.039525691699604744, 0.0321285140562249, 0.072, 0.10236220472440945, 0.07749077490774908, 0.1626984126984127, 0.045081967213114756, 0.12903225806451613, 0.12075471698113208, 0.1593625498007968, 0.16535433070866143, 0.056818181818181816, 0.18930041152263374, 0.13740458015267176, 0.08860759493670886, 0.08, 0.26907630522088355, 0.13793103448275862, 0.18333333333333332, 0.15671641791044777, 0.11952191235059761, 0.03278688524590164, 0.15503875968992248, 0.23622047244094488, 0.17063492063492064, 0.25806451612903225, 0.1947565543071161, 0.12030075187969924]

# average = [0.03, -0.01, -0.03, -0.06, -0.17, 0.03, 0.11, -0.05, 0.13, 0.12, 0.04, 0.04, 0.07, -0.03, 0.2, -0.06, 0.03, 0.12, 0.05, 0.01, -0.05, 0.13, -0.19, -0.13, 0.03, 0.03, -0.08, 0.14, -0.05, -0.14, -0.01, 0.13, -0.15, 0.05, -0.01, -0.01, -0.22, 0.06, 0.01, 0.05, -0.09, 0.04, 0.02, -0.02, -0.09, -0.02, 0.04, -0.03, -0.09, -0.01, 0.07, 0.06, -0.18, 0.07, -0.04, 0.06, 0.15, 0.02, -0.11, -0.19, 0.0, -0.06, -0.07, -0.02, 0.41, -0.01, -0.17, 0.05, -0.05, -0.03, 0.16, -0.12, -0.05, 0.01, 0.13, -0.09, -0.03, -0.26, 0.07, 0.04, 0.0, 0.02, -0.03, 0.01, 0.1, 0.02, -0.02, 0.03, 0.21, 0.07, -0.11, 0.02, 0.06, 0.02, 0.16, -0.04, 0.09, 0.04, 0.18, 0.02, 0.11, -0.03, 0.14, 0.03, 0.08, -0.24, 0.02, -0.08, 0.03, -0.02, 0.08, 0.03, 0.07, -0.09, 0.23, -0.01, -0.07, -0.06, 0.06, 0.06, 0.08, -0.09, -0.14]

# townie = [0.3611111111111111, 0.18867924528301888, 0.15254237288135594, 0.06, -0.13333333333333333, 0.375, 0.2222222222222222, 0.057692307692307696, 0.2909090909090909, 0.2962962962962963, 0.26666666666666666, 0.021739130434782608, 0.25, 0.05660377358490566, 0.2857142857142857, 0.16981132075471697, 0.2391304347826087, 0.36585365853658536, 0.30612244897959184, 0.4, 0.17647058823529413, 0.38, -0.12, 0.07692307692307693, 0.17391304347826086, 0.3018867924528302, 0.14545454545454545, 0.25862068965517243, 0.018518518518518517, 0.20408163265306123, -0.0784313725490196, 0.4782608695652174, 0.10638297872340426, 0.18518518518518517, 0.07317073170731707, 0.18867924528301888, 0.1016949152542373, 0.34782608695652173, 0.04, 0.061224489795918366, 0.09259259259259259, 0.13636363636363635, 0.16363636363636364, 0.21052631578947367, 0.16, 0.2692307692307692, 0.3191489361702128, 0.18867924528301888, 0.14893617021276595, 0.0851063829787234, 0.22448979591836735, 0.3, -0.04081632653061224, 0.20833333333333334, 0.3111111111111111, 0.17647058823529413, 0.375, 0.1724137931034483, 0.23404255319148937, -0.11904761904761904, 0.37209302325581395, 0.08, -0.11538461538461539, 0.07407407407407407, 0.5370370370370371, 0.1568627450980392, 0.06382978723404255, 0.30434782608695654, 0.10869565217391304, 0.05555555555555555, 0.3898305084745763, 0.18518518518518517, 0.1724137931034483, 0.06779661016949153, 0.25925925925925924, 0.1694915254237288, 0.1, -0.019230769230769232, 0.39215686274509803, 0.3125, 0.1875, 0.125, 0.13043478260869565, 0.3148148148148148, 0.21311475409836064, 0.09433962264150944, 0.2127659574468085, 0.15384615384615385, 0.45614035087719296, 0.3392857142857143, 0.07692307692307693, 0.25, 0.21568627450980393, 0.2777777777777778, 0.18867924528301888, 0.2553191489361702, 0.37254901960784315, 0.09433962264150944, 0.44, 0.19148936170212766, 0.3333333333333333, 0.06976744186046512, 0.40816326530612246, 0.3958333333333333, 0.29545454545454547, -0.1836734693877551, 0.25925925925925924, 0.32608695652173914, 0.2765957446808511, 0.1509433962264151, 0.20833333333333334, 0.2708333333333333, 0.125, 0.22727272727272727, 0.5, 0.17391304347826086, 0.32558139534883723, 0.13043478260869565, 0.2545454545454545, 0.15555555555555556, 0.32653061224489793, 0.10204081632653061, 0.0]

# mafia = [-0.15625, -0.23404255319148937, -0.2926829268292683, -0.18, -0.2, -0.28846153846153844, -0.021739130434782608, -0.16666666666666666, -0.06666666666666667, -0.08695652173913043, -0.14545454545454545, 0.05555555555555555, -0.09615384615384616, -0.1276595744680851, 0.11764705882352941, -0.3191489361702128, -0.14814814814814814, -0.05084745762711865, -0.19607843137254902, -0.3090909090909091, -0.2857142857142857, -0.12, -0.26, -0.3541666666666667, -0.09259259259259259, -0.2765957446808511, -0.35555555555555557, -0.023809523809523808, -0.13043478260869565, -0.47058823529411764, 0.061224489795918366, -0.16666666666666666, -0.37735849056603776, -0.10869565217391304, -0.06779661016949153, -0.23404255319148937, -0.6829268292682927, -0.18518518518518517, -0.02, 0.0392156862745098, -0.30434782608695654, -0.03571428571428571, -0.15555555555555556, -0.32558139534883723, -0.34, -0.3333333333333333, -0.20754716981132076, -0.2765957446808511, -0.3018867924528302, -0.09433962264150944, -0.0784313725490196, -0.18, -0.3137254901960784, -0.057692307692307696, -0.32727272727272727, -0.061224489795918366, -0.057692307692307696, -0.19047619047619047, -0.41509433962264153, -0.2413793103448276, -0.2807017543859649, -0.2, -0.020833333333333332, -0.13043478260869565, 0.2608695652173913, -0.1836734693877551, -0.37735849056603776, -0.16666666666666666, -0.18518518518518517, -0.078125, -0.17073170731707318, -0.4782608695652174, -0.35714285714285715, -0.07317073170731707, -0.021739130434782608, -0.4634146341463415, -0.16, -0.5208333333333334, -0.2653061224489796, -0.21153846153846154, -0.17307692307692307, -0.07692307692307693, -0.16666666666666666, -0.34782608695652173, -0.07692307692307693, -0.06382978723404255, -0.22641509433962265, -0.10416666666666667, -0.11627906976744186, -0.2727272727272727, -0.3125, -0.16071428571428573, -0.10204081632653061, -0.2826086956521739, 0.1276595744680851, -0.3018867924528302, -0.20408163265306123, -0.02127659574468085, -0.08, -0.1320754716981132, -0.12244897959183673, -0.10526315789473684, -0.11764705882352941, -0.3076923076923077, -0.08928571428571429, -0.29411764705882354, -0.2608695652173913, -0.42592592592592593, -0.18867924528301888, -0.2127659574468085, -0.038461538461538464, -0.19230769230769232, 0.0, -0.3392857142857143, 0.017857142857142856, -0.16666666666666666, -0.3684210526315789, -0.2222222222222222, -0.17777777777777778, -0.01818181818181818, -0.1568627450980392, -0.27450980392156865, -0.2153846153846154]

x = range(len(townie))

fig = plt.figure()

plt.scatter(x, townie, s=10, label='Average reward when on the town team')
plt.plot(x, townie)
plt.scatter(x, mafia, s=10, label='Average reward when on the mafia team')
plt.plot(x, mafia)
plt.scatter(x, average, s=10, label='Average reward on either team')
plt.plot(x, average)
plt.legend(loc='best')
plt.xlabel("Iterations (Thousands)")
plt.ylabel("Rewards (1 for winning, -1 for losing)")
#plt.show()
plt.savefig('randomagent.png')