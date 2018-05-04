import matplotlib.pyplot as plt
import numpy as np

exploitability = [[0.5000000149011612, 0.4978290945291519, 0.46319785714149475, 0.4276937395334244, 0.39418716728687286, 0.3636068180203438, 0.33666350692510605, 0.3130555897951126, 0.29132218658924103, 0.27191678434610367, 0.2545613422989845, 0.24015242606401443, 0.2272854596376419, 0.21509356796741486, 0.22063414752483368, 0.23371243476867676, 0.24148300290107727, 0.24458829313516617, 0.24423546344041824, 0.24155881255865097, 0.2370259389281273, 0.23060782998800278, 0.22245362401008606, 0.21361251920461655, 0.20617496967315674, 0.19738195836544037, 0.18819766491651535, 0.1795748770236969, 0.17045891284942627, 0.16233938932418823, 0.15509384870529175, 0.1470349133014679, 0.1394399106502533, 0.13367360085248947, 0.1275649517774582, 0.12196613848209381, 0.11749789118766785, 0.11361652612686157, 0.10947418212890625, 0.10577919334173203, 0.10248218476772308, 0.0991998016834259, 0.09636807441711426, 0.09384617209434509, 0.09144116193056107, 0.08963967859745026, 0.08763778209686279, 0.08539438992738724, 0.08346189558506012, 0.081619992852211, 0.08030200004577637, 0.0780101865530014, 0.07543810456991196, 0.0730157420039177, 0.0704546645283699, 0.06721077114343643, 0.06334500759840012, 0.059408485889434814, 0.05469755083322525, 0.04992459714412689, 0.04540840536355972, 0.04042947292327881, 0.034722767770290375, 0.0307009220123291, 0.02862328290939331, 0.02741473913192749, 0.02461904287338257, 0.02376246452331543, 0.024513423442840576, 0.027452945709228516, 0.027064204216003418, 0.027215659618377686, 0.02370607852935791, 0.024731099605560303, 0.03148925304412842, 0.04090353846549988, 0.047418057918548584, 0.05332234501838684, 0.05842089653015137, 0.06268975138664246, 0.06771627068519592, 0.07179316878318787, 0.07501161098480225, 0.07715022563934326, 0.0811852514743805, 0.08458977937698364, 0.08814305067062378, 0.09173107147216797, 0.09238240122795105, 0.09469792246818542, 0.09647810459136963, 0.09918388724327087, 0.1007479876279831, 0.10123708844184875, 0.10387973487377167, 0.10306316614151001, 0.10324366390705109, 0.1039019376039505, 0.10489004850387573, 0.10540622472763062, 0.10533271729946136, 0.10474447906017303, 0.10386107861995697, 0.10072100162506104, 0.09935913980007172, 0.09735710918903351, 0.09502191841602325, 0.09171219170093536, 0.08879929780960083, 0.0851229876279831, 0.08258050680160522, 0.07890559732913971, 0.07545675337314606, 0.07164549827575684, 0.06842342019081116, 0.06498104333877563, 0.06127670407295227, 0.057201460003852844, 0.053785428404808044, 0.04953943192958832, 0.04394546151161194, 0.03984488546848297, 0.04022660851478577, 0.04065205156803131, 0.04082395136356354, 0.0400547981262207, 0.03964069485664368, 0.03985010087490082, 0.040056899189949036, 0.040069326758384705, 0.0393599271774292, 0.039411187171936035, 0.03868992626667023, 0.03785070776939392, 0.03814345598220825, 0.03770306706428528, 0.037467941641807556, 0.03695414960384369, 0.03677397966384888, 0.03652045130729675, 0.036653071641922, 0.036428071558475494, 0.036215007305145264, 0.036183491349220276, 0.03546391427516937, 0.03515617549419403, 0.03499762713909149, 0.035220809280872345, 0.034927159547805786, 0.03454749286174774, 0.03495384007692337, 0.03521531820297241, 0.035295650362968445, 0.03524384647607803, 0.03551331162452698, 0.03606916218996048, 0.036390095949172974, 0.036620981991291046, 0.036592788994312286, 0.03664568066596985, 0.03717547655105591, 0.037942349910736084, 0.038156166672706604, 0.0388193354010582, 0.03895927965641022, 0.03920971602201462, 0.04009982943534851, 0.04041113704442978, 0.041090935468673706, 0.04202199727296829, 0.042317941784858704, 0.04290667176246643, 0.04360023885965347, 0.043646566569805145, 0.04405965656042099, 0.04413797706365585, 0.04486645758152008, 0.04509872943162918, 0.04534347355365753, 0.045482754707336426, 0.04589737951755524, 0.04602694511413574, 0.04581809788942337, 0.045995764434337616, 0.04664991796016693, 0.04687482863664627, 0.047144703567028046, 0.04744783788919449, 0.04778595268726349, 0.0483538955450058, 0.048409342765808105, 0.04819769412279129, 0.04864353686571121, 0.04800591617822647, 0.04776838421821594, 0.046942539513111115, 0.046848803758621216, 0.047042421996593475, 0.04674631357192993, 0.0465017706155777], [0.5000000149011612, 0.49805350601673126, 0.46883171796798706, 0.43844856321811676, 0.40649496018886566, 0.37680237740278244, 0.3507995903491974, 0.32743337005376816, 0.3065115064382553, 0.2873184457421303, 0.2700977325439453, 0.254355289041996, 0.24051421880722046, 0.2287791222333908, 0.21797680109739304, 0.22171394526958466, 0.22578318417072296, 0.2258496880531311, 0.22462156414985657, 0.22044821083545685, 0.21457475423812866, 0.2075662761926651, 0.20010724663734436, 0.19122925400733948, 0.1825549155473709, 0.17360810935497284, 0.16427722573280334, 0.1558120921254158, 0.14823296666145325, 0.14172733575105667, 0.13470151275396347, 0.12810827046632767, 0.12166325747966766, 0.11657285690307617, 0.11192087084054947, 0.10804259777069092, 0.10428716987371445, 0.1002989187836647, 0.09785281121730804, 0.09497124701738358, 0.09225854277610779, 0.0892288014292717, 0.08627236634492874, 0.08394008874893188, 0.08174379914999008, 0.07970154285430908, 0.07770252227783203, 0.07597000151872635, 0.07436762005090714, 0.07295771688222885, 0.0718379020690918, 0.06967510282993317, 0.06832605600357056, 0.07446348667144775, 0.07568144798278809, 0.07848262786865234, 0.07865661382675171, 0.07423412799835205, 0.06866782903671265, 0.06201756000518799, 0.05308818817138672, 0.04676070809364319, 0.042821213603019714, 0.03877892345190048, 0.03363867849111557, 0.02945655584335327, 0.02675330638885498, 0.02236872911453247, 0.018703937530517578, 0.014845490455627441, 0.010191738605499268, 0.005645975470542908, 0.0028356611728668213, 0.0057048499584198, 0.014828085899353027, 0.022204041481018066, 0.0320872962474823, 0.03932690620422363, 0.04713761806488037, 0.052761346101760864, 0.057815998792648315, 0.06407144665718079, 0.06855243444442749, 0.07159951329231262, 0.07672533392906189, 0.08142295479774475, 0.08538022637367249, 0.08847801387310028, 0.09097142517566681, 0.09246519207954407, 0.09444227814674377, 0.09480208158493042, 0.09590047597885132, 0.09694936871528625, 0.09935514628887177, 0.10164359211921692, 0.1018393486738205, 0.10262712836265564, 0.1036277562379837, 0.10390506684780121, 0.10314382612705231, 0.10237161815166473, 0.1017463356256485, 0.1000310629606247, 0.09897780418395996, 0.09882232546806335, 0.0989895612001419, 0.09924879670143127, 0.09849020838737488, 0.09574997425079346, 0.09486284852027893, 0.09298187494277954, 0.09049327671527863, 0.08836056292057037, 0.08649526536464691, 0.08430737257003784, 0.08369000256061554, 0.08522048592567444, 0.08558197319507599, 0.08536574244499207, 0.08585414290428162, 0.08563198149204254, 0.08630599081516266, 0.08572840690612793, 0.08572320640087128, 0.08556996285915375, 0.08429610729217529, 0.08383502066135406, 0.08404520153999329, 0.08237104117870331, 0.08105964958667755, 0.08083763718605042, 0.08006571233272552, 0.07947883009910583, 0.07883298397064209, 0.07812802493572235, 0.07818835973739624, 0.07834945619106293, 0.07829290628433228, 0.07780963182449341, 0.07696812599897385, 0.07632700353860855, 0.07565261423587799, 0.07433964312076569, 0.07397876679897308, 0.07277961075305939, 0.07236867398023605, 0.07123830914497375, 0.06945375353097916, 0.06796976923942566, 0.06705382466316223, 0.06675677746534348, 0.06578744202852249, 0.06528770923614502, 0.06525513529777527, 0.06453265994787216, 0.06284963339567184, 0.061598554253578186, 0.061400189995765686, 0.059808701276779175, 0.0588371604681015, 0.058090940117836, 0.05656972527503967, 0.055457375943660736, 0.05446694791316986, 0.052894845604896545, 0.05188616365194321, 0.05066873878240585, 0.04836773872375488, 0.04772216081619263, 0.04578389227390289, 0.043859824538230896, 0.04256246238946915, 0.040984369814395905, 0.03984525799751282, 0.038450613617897034, 0.03708715736865997, 0.03551085293292999, 0.03383520245552063, 0.03274238109588623, 0.030676275491714478, 0.02912139892578125, 0.028063803911209106, 0.026696547865867615, 0.025516048073768616, 0.024178847670555115, 0.022897973656654358, 0.021787315607070923, 0.021341517567634583, 0.019967809319496155, 0.018845632672309875, 0.018125131726264954, 0.017606958746910095, 0.01745210587978363, 0.016087472438812256, 0.01529550552368164, 0.014598965644836426, 0.01368163526058197, 0.01278531551361084, 0.01211819052696228]]# townie = [0.3611111111111111, 0.18867924528301888, 0.15254237288135594, 0.06, -0.13333333333333333, 0.375, 0.2222222222222222, 0.057692307692307696, 0.2909090909090909, 0.2962962962962963, 0.26666666666666666, 0.021739130434782608, 0.25, 0.05660377358490566, 0.2857142857142857, 0.16981132075471697, 0.2391304347826087, 0.36585365853658536, 0.30612244897959184, 0.4, 0.17647058823529413, 0.38, -0.12, 0.07692307692307693, 0.17391304347826086, 0.3018867924528302, 0.14545454545454545, 0.25862068965517243, 0.018518518518518517, 0.20408163265306123, -0.0784313725490196, 0.4782608695652174, 0.10638297872340426, 0.18518518518518517, 0.07317073170731707, 0.18867924528301888, 0.1016949152542373, 0.34782608695652173, 0.04, 0.061224489795918366, 0.09259259259259259, 0.13636363636363635, 0.16363636363636364, 0.21052631578947367, 0.16, 0.2692307692307692, 0.3191489361702128, 0.18867924528301888, 0.14893617021276595, 0.0851063829787234, 0.22448979591836735, 0.3, -0.04081632653061224, 0.20833333333333334, 0.3111111111111111, 0.17647058823529413, 0.375, 0.1724137931034483, 0.23404255319148937, -0.11904761904761904, 0.37209302325581395, 0.08, -0.11538461538461539, 0.07407407407407407, 0.5370370370370371, 0.1568627450980392, 0.06382978723404255, 0.30434782608695654, 0.10869565217391304, 0.05555555555555555, 0.3898305084745763, 0.18518518518518517, 0.1724137931034483, 0.06779661016949153, 0.25925925925925924, 0.1694915254237288, 0.1, -0.019230769230769232, 0.39215686274509803, 0.3125, 0.1875, 0.125, 0.13043478260869565, 0.3148148148148148, 0.21311475409836064, 0.09433962264150944, 0.2127659574468085, 0.15384615384615385, 0.45614035087719296, 0.3392857142857143, 0.07692307692307693, 0.25, 0.21568627450980393, 0.2777777777777778, 0.18867924528301888, 0.2553191489361702, 0.37254901960784315, 0.09433962264150944, 0.44, 0.19148936170212766, 0.3333333333333333, 0.06976744186046512, 0.40816326530612246, 0.3958333333333333, 0.29545454545454547, -0.1836734693877551, 0.25925925925925924, 0.32608695652173914, 0.2765957446808511, 0.1509433962264151, 0.20833333333333334, 0.2708333333333333, 0.125, 0.22727272727272727, 0.5, 0.17391304347826086, 0.32558139534883723, 0.13043478260869565, 0.2545454545454545, 0.15555555555555556, 0.32653061224489793, 0.10204081632653061, 0.0]
# exploitability = [[0.5000000149011612, 0.5000000149011612, 0.46562159061431885, 0.42933525145053864, 0.3968251049518585, 0.366272397339344, 0.33849626779556274, 0.31437212973833084, 0.29176176339387894, 0.27278701961040497, 0.2556282579898834, 0.24053265899419785, 0.22704128921031952, 0.21509935706853867, 0.22351188957691193, 0.23113074898719788, 0.23398856818675995, 0.2365504652261734, 0.2370726466178894, 0.23464945703744888, 0.22967764735221863, 0.2233787402510643, 0.214931420981884, 0.20628871023654938, 0.19802667945623398, 0.1886027455329895, 0.17986410856246948, 0.17146428674459457, 0.16317344456911087, 0.15574514865875244, 0.14830559492111206, 0.14143520593643188, 0.13484130054712296, 0.12834611535072327, 0.12392197549343109, 0.11807489395141602, 0.11344585567712784, 0.10864473879337311, 0.10521907359361649, 0.10218814760446548, 0.09911476820707321, 0.09672310203313828, 0.09345763176679611, 0.09017401933670044, 0.08830244094133377, 0.08590614050626755, 0.08346742391586304, 0.08075253665447235, 0.07879848778247833, 0.0764775425195694, 0.07434925436973572, 0.07191314548254013, 0.0698971077799797, 0.06704769283533096, 0.06382174789905548, 0.06012871116399765, 0.059335410594940186, 0.0638885498046875, 0.06281423568725586, 0.061580002307891846, 0.06291687488555908, 0.06335693597793579, 0.06255179643630981, 0.06017822027206421, 0.05829334259033203, 0.05967724323272705, 0.05553388595581055, 0.055377423763275146, 0.051258623600006104, 0.04942077398300171, 0.04738497734069824, 0.04700571298599243, 0.042357027530670166, 0.05253925919532776, 0.061359286308288574, 0.06851077079772949, 0.07419326901435852, 0.07955831289291382, 0.08521389961242676, 0.09096848964691162, 0.09666678309440613, 0.099418044090271, 0.10434532165527344, 0.10813900828361511, 0.11236193776130676, 0.11450785398483276, 0.11793091893196106, 0.12017396092414856, 0.12095749378204346, 0.12376901507377625, 0.12551221251487732, 0.12807703018188477, 0.12834055721759796, 0.12885509431362152, 0.12908215820789337, 0.1286860555410385, 0.12832584977149963, 0.12897822260856628, 0.13029389083385468, 0.12849102914333344, 0.12837503850460052, 0.12725664675235748, 0.12809719145298004, 0.1288156807422638, 0.12821215391159058, 0.12673388421535492, 0.12608303129673004, 0.12593534588813782, 0.12484711408615112, 0.12440526485443115, 0.12342245876789093, 0.12204490602016449, 0.11969876289367676, 0.11869484186172485, 0.11667540669441223, 0.11434806883335114, 0.11257441341876984, 0.11023128032684326, 0.10879115760326385, 0.10761243104934692, 0.10582156479358673, 0.10261575877666473, 0.10041828453540802, 0.09815886616706848, 0.09553156793117523, 0.09408575296401978, 0.09020741283893585, 0.08717100322246552, 0.08436742424964905, 0.08324752748012543, 0.08341629803180695, 0.08403243124485016, 0.08334483206272125, 0.08299718797206879, 0.08199575543403625, 0.08178453147411346, 0.08087384700775146, 0.08004896342754364, 0.07925516366958618, 0.07996787130832672, 0.0789848119020462, 0.07875379920005798, 0.07851952314376831, 0.07883743941783905, 0.07868552207946777, 0.07801350951194763, 0.07663759589195251, 0.0760522186756134, 0.07543614506721497, 0.07543601095676422, 0.0747944563627243, 0.07475735247135162, 0.07403971254825592, 0.07281652092933655, 0.07261598855257034, 0.07258200645446777, 0.07272091507911682, 0.07183825224637985, 0.07160791754722595, 0.07135817408561707, 0.07106366008520126, 0.07027201354503632, 0.06974325329065323, 0.06912881880998611, 0.06905852258205414, 0.06836418807506561, 0.06825648993253708, 0.06778901815414429, 0.06700506806373596, 0.06631965190172195, 0.06583679467439651, 0.06524798274040222, 0.06517151743173599, 0.06449843943119049, 0.06442681699991226, 0.06347978860139847, 0.06354688853025436, 0.06289135664701462, 0.062475040555000305, 0.062446169555187225, 0.06209775060415268, 0.06201622635126114, 0.06155663728713989, 0.06119450926780701, 0.060447581112384796, 0.059932321310043335, 0.059092894196510315, 0.05876964330673218, 0.058082498610019684, 0.05805119872093201, 0.057964473962783813, 0.05796973407268524, 0.057265378534793854, 0.056805990636348724, 0.0568971186876297, 0.0569600909948349, 0.05696738511323929, 0.05658233165740967, 0.055993832647800446, 0.055843450129032135], [0.5000000149011612, 0.5000000149011612, 0.4694581925868988, 0.43509089946746826, 0.40090933442115784, 0.3713221177458763, 0.34578870236873627, 0.3219914510846138, 0.2999863028526306, 0.2808249518275261, 0.26422373950481415, 0.24928997457027435, 0.2356875315308571, 0.22378641366958618, 0.21481165289878845, 0.21909715235233307, 0.21972136199474335, 0.21748289465904236, 0.2136029750108719, 0.20742125809192657, 0.19983582198619843, 0.19225679337978363, 0.18296220898628235, 0.1736956089735031, 0.16521286964416504, 0.15699951350688934, 0.14871954917907715, 0.1410491019487381, 0.1343385949730873, 0.1285812109708786, 0.1225062757730484, 0.11630672216415405, 0.11174676567316055, 0.10666222125291824, 0.1031787246465683, 0.09856948256492615, 0.09508398175239563, 0.09163151681423187, 0.08823786675930023, 0.08571875095367432, 0.08317900449037552, 0.08102243393659592, 0.07836739718914032, 0.0761963352560997, 0.07423552125692368, 0.0731608122587204, 0.07196252048015594, 0.0708327665925026, 0.06956275552511215, 0.0697784423828125, 0.08329653739929199, 0.09508752822875977, 0.1008613109588623, 0.10741102695465088, 0.10817837715148926, 0.10992234945297241, 0.10677766799926758, 0.10442239046096802, 0.10076624155044556, 0.09266161918640137, 0.08579552173614502, 0.08299088478088379, 0.07833027839660645, 0.06866872310638428, 0.06030553579330444, 0.05152451992034912, 0.04273402690887451, 0.03171122074127197, 0.02560138702392578, 0.01929175853729248, 0.013690710067749023, 0.005806922912597656, 0.005909711122512817, 0.0170251727104187, 0.027890443801879883, 0.04017752408981323, 0.05082857608795166, 0.06039056181907654, 0.06934705376625061, 0.07727283239364624, 0.08370155096054077, 0.09040454030036926, 0.098082035779953, 0.10454583168029785, 0.10928849875926971, 0.11452895402908325, 0.11710451543331146, 0.12142692506313324, 0.12567651271820068, 0.12908044457435608, 0.1321338415145874, 0.13621094822883606, 0.1375717967748642, 0.13811518251895905, 0.1389661729335785, 0.1391019970178604, 0.1394629031419754, 0.13906759023666382, 0.13900813460350037, 0.1401602029800415, 0.14113669097423553, 0.1420285552740097, 0.1407618671655655, 0.13976362347602844, 0.13896265625953674, 0.1369745433330536, 0.1375424861907959, 0.13763588666915894, 0.13735733926296234, 0.1385546177625656, 0.1381351202726364, 0.13656124472618103, 0.13503152132034302, 0.1330517679452896, 0.13124191761016846, 0.12960122525691986, 0.12827074527740479, 0.12657563388347626, 0.12471702694892883, 0.12212596833705902, 0.12074530124664307, 0.1179763525724411, 0.11545595526695251, 0.1126878559589386, 0.11008206009864807, 0.10823288559913635, 0.10590408742427826, 0.10229092836380005, 0.09932693839073181, 0.09903021156787872, 0.09751328825950623, 0.0965278297662735, 0.09593792259693146, 0.09529155492782593, 0.09532202780246735, 0.09442088007926941, 0.09358422458171844, 0.09285514056682587, 0.09231345355510712, 0.09136980772018433, 0.09042046964168549, 0.08925488591194153, 0.08887369930744171, 0.08840532600879669, 0.08861072361469269, 0.08755810558795929, 0.08687594532966614, 0.08704990148544312, 0.0864868015050888, 0.08586037904024124, 0.08467307686805725, 0.08401212096214294, 0.08354011178016663, 0.08319147676229477, 0.08243127912282944, 0.08183346688747406, 0.08123103529214859, 0.08025841414928436, 0.0801110491156578, 0.07971419394016266, 0.07957708835601807, 0.07962841540575027, 0.07912801206111908, 0.07800397276878357, 0.07717924565076828, 0.07685164362192154, 0.07631557434797287, 0.07615835219621658, 0.0756588876247406, 0.07471363246440887, 0.07456264644861221, 0.07422417402267456, 0.07328716665506363, 0.07266827672719955, 0.07191269099712372, 0.07188811898231506, 0.07237950712442398, 0.0718705803155899, 0.07093499600887299, 0.07065900415182114, 0.06959515064954758, 0.06896782666444778, 0.06840917468070984, 0.06795839220285416, 0.06778766959905624, 0.06709029525518417, 0.06679895520210266, 0.06639888882637024, 0.06559614092111588, 0.06511606276035309, 0.06455951929092407, 0.06352704018354416, 0.0634235143661499, 0.06309136003255844, 0.06248491257429123, 0.061632491648197174, 0.061617374420166016, 0.06129568815231323, 0.06079976260662079, 0.06097225099802017]]
# mafia = [-0.15625, -0.23404255319148937, -0.2926829268292683, -0.18, -0.2, -0.28846153846153844, -0.021739130434782608, -0.16666666666666666, -0.06666666666666667, -0.08695652173913043, -0.14545454545454545, 0.05555555555555555, -0.09615384615384616, -0.1276595744680851, 0.11764705882352941, -0.3191489361702128, -0.14814814814814814, -0.05084745762711865, -0.19607843137254902, -0.3090909090909091, -0.2857142857142857, -0.12, -0.26, -0.3541666666666667, -0.09259259259259259, -0.2765957446808511, -0.35555555555555557, -0.023809523809523808, -0.13043478260869565, -0.47058823529411764, 0.061224489795918366, -0.16666666666666666, -0.37735849056603776, -0.10869565217391304, -0.06779661016949153, -0.23404255319148937, -0.6829268292682927, -0.18518518518518517, -0.02, 0.0392156862745098, -0.30434782608695654, -0.03571428571428571, -0.15555555555555556, -0.32558139534883723, -0.34, -0.3333333333333333, -0.20754716981132076, -0.2765957446808511, -0.3018867924528302, -0.09433962264150944, -0.0784313725490196, -0.18, -0.3137254901960784, -0.057692307692307696, -0.32727272727272727, -0.061224489795918366, -0.057692307692307696, -0.19047619047619047, -0.41509433962264153, -0.2413793103448276, -0.2807017543859649, -0.2, -0.020833333333333332, -0.13043478260869565, 0.2608695652173913, -0.1836734693877551, -0.37735849056603776, -0.16666666666666666, -0.18518518518518517, -0.078125, -0.17073170731707318, -0.4782608695652174, -0.35714285714285715, -0.07317073170731707, -0.021739130434782608, -0.4634146341463415, -0.16, -0.5208333333333334, -0.2653061224489796, -0.21153846153846154, -0.17307692307692307, -0.07692307692307693, -0.16666666666666666, -0.34782608695652173, -0.07692307692307693, -0.06382978723404255, -0.22641509433962265, -0.10416666666666667, -0.11627906976744186, -0.2727272727272727, -0.3125, -0.16071428571428573, -0.10204081632653061, -0.2826086956521739, 0.1276595744680851, -0.3018867924528302, -0.20408163265306123, -0.02127659574468085, -0.08, -0.1320754716981132, -0.12244897959183673, -0.10526315789473684, -0.11764705882352941, -0.3076923076923077, -0.08928571428571429, -0.29411764705882354, -0.2608695652173913, -0.42592592592592593, -0.18867924528301888, -0.2127659574468085, -0.038461538461538464, -0.19230769230769232, 0.0, -0.3392857142857143, 0.017857142857142856, -0.16666666666666666, -0.3684210526315789, -0.2222222222222222, -0.17777777777777778, -0.01818181818181818, -0.1568627450980392, -0.27450980392156865, -0.2153846153846154]

x = range(len(exploitability[0]))

fig = plt.figure()

plt.scatter(x, exploitability[0], s=10, label='Average reward when on the town team')
plt.plot(x, exploitability[0])
plt.scatter(x, exploitability[1], s=10, label='Average reward when on the mafia team')
plt.plot(x, exploitability[1])
plt.legend(loc='best')
plt.xlabel("Iterations (Thousands)")
plt.ylabel("exploitability")
#plt.show()
plt.savefig('randomagent.png')