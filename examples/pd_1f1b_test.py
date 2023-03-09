import matplotlib.pyplot as plt

from rene import InstructionDAG, Synchronous1F1B, Forward, Backward, PipelineVisualizer


# Instantiate the Instruction DAG.
dag = InstructionDAG(
    schedule_type=Synchronous1F1B,
    num_stages=4,
    num_micro_batches=2,
    # F-0 y=-2x+60 F-1 y=-x+50 B-0 y=-3x+80 B-1 y=-4x+100
    # time_costs={Forward: {0: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)],
    # 1: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)], 2: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)],
    # 3: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)]}, 
    # Backward: {0: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    # 1: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)], 2: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    # 3: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)]}},
    # time_costs={Forward: {0: [(10, 40, 1000), (20, 20, 400), (15, 30, 700)],
    # 1: [(10, 40, 1000), (30, 20, 500), (20, 30, 700)]}, 
    # Backward: {0: [(10, 50, 1000), (20, 20, 500), (15, 35, 700)],
    # 1: [(10, 60, 1000), (20, 20, 400), (15, 40, 700)]}},
    time_costs={Forward:{0:[(0.4263051748275757, 83.29230958323426, 1740), (0.4330758889516194, 82.94802923479159, 1515), (0.4358434279759725, 82.61222014040194, 1485), (0.4371887604395548, 81.30966310095836, 1470), (0.4386793613433838, 81.11103748516511, 1455), (0.4401282072067261, 79.56087291090597, 1440), (0.4419130643208822, 78.7731771020062, 1425), (0.4432432730992635, 78.63421705229266, 1410), (0.4448753197987874, 77.84463170364691, 1395), (0.4464127858479818, 77.69006826686308, 1380), (0.4480388005574544, 77.52166194669755, 1365), (0.4497989257176717, 76.49666033933677, 1350), (0.4514543374379476, 75.52023996612489, 1335), (0.4532374382019043, 75.15532929425757, 1320), (0.4549808979034423, 74.784988119022, 1305), (0.4568286577860514, 73.63920951792879, 1290), (0.4586692571640015, 72.79607310579468, 1275), (0.462383500734965, 72.69142411683018, 1245), (0.4645944436391194, 71.48248669359062, 1230), (0.4668869495391846, 70.88370664517504, 1215), (0.4691606124242147, 68.4178979402274, 1200), (0.4740860939025879, 68.25152567754404, 1170), (0.4767746766408284, 67.53369924367894, 1155), (0.4792649825414022, 66.45805476955643, 1140)], 1:[(0.4253450314203898, 82.39996391744702, 1740), (0.4247862974802653, 82.97753336002354, 1725), (0.4246158281962077, 86.30517905367867, 1665), (0.4381737152735392, 80.8923439957847, 1425), (0.4398127794265747, 79.98670670983644, 1410), (0.4413801272710164, 79.7085352304648, 1395), (0.4429271141688028, 78.72959941597465, 1380), (0.444599707921346, 78.42939495863078, 1365), (0.4463875770568847, 77.1556251210402, 1350), (0.4479538679122924, 76.05767328373423, 1335), (0.4497070471445719, 75.62383747624074, 1320), (0.4515349308649699, 75.20328333818169, 1305), (0.4534603754679362, 73.96682779923783, 1290), (0.4551204999287923, 73.52966539312933, 1275), (0.4570103327433268, 72.27848545833703, 1260), (0.461008874575297, 71.48763266028118, 1230), (0.4632518609364827, 70.88982192514759, 1215), (0.4655507405598958, 70.47279454358419, 1200), (0.4679613351821899, 69.78133712452275, 1185), (0.4706073999404907, 68.69301617634783, 1170), (0.4731255531311035, 67.41562580540403, 1155), (0.4756110509236653, 67.08901802449861, 1140), (0.4782854318618774, 66.47914540797562, 1125), (0.4811039288838704, 65.900544831032, 1110), (0.4867759307225545, 65.79223893185763, 1080), (0.4898956775665283, 65.78876332575119, 1065), (0.5063872734705607, 65.1291889044789, 990)], 2:[(0.4233403046925862, 81.3169321777604, 1740), (0.422809108098348, 81.84671147125891, 1725), (0.4298927386601766, 81.27479263794835, 1515), (0.4312768459320068, 80.72147717376049, 1500), (0.4328443209330241, 80.35094444013265, 1485), (0.4339702765146891, 79.01455129072369, 1470), (0.4354534387588501, 77.85841368042936, 1455), (0.4371157725652059, 77.20486759900808, 1440), (0.4384678602218628, 75.5893038034881, 1425), (0.4420030196507772, 73.90790747146478, 1395), (0.4434053977330526, 73.70787559007482, 1380), (0.4450965881347656, 73.6138291575254, 1365), (0.4466559648513794, 73.23684341188537, 1350), (0.4482715209325155, 72.15816373751292, 1335), (0.4500241835912069, 71.8355620776027, 1320), (0.4515429178873698, 71.63251097560904, 1305), (0.4534940083821615, 70.48789897331514, 1290), (0.4556108792622884, 68.96325250797605, 1275), (0.4593333005905151, 67.74702651752037, 1245), (0.4617045561472574, 67.5763329126193, 1230), (0.4655513445536295, 66.02412156221472, 1200), (0.4680305083592733, 65.45142500607386, 1185), (0.4704320271809896, 65.08943769135496, 1170), (0.4733426968256632, 64.46983685257939, 1155), (0.4759677251180013, 64.19308880085947, 1140), (0.4783496538798014, 64.1082981673961, 1125), (0.4809296687444051, 63.54811481088959, 1110), (0.4867011388142904, 63.521086967734334, 1080), (0.492807920773824, 63.078602215969326, 1050)], 3:[(0.5384431838989258, 105.93321218412797, 1740), (0.5587270259857178, 104.69608880679905, 1410), (0.5625051657358805, 103.22631220535624, 1380), (0.564973521232605, 102.64122201764349, 1365), (0.5670456568400065, 100.6022778565927, 1350), (0.5691642999649048, 98.21605655426077, 1335), (0.5716225624084472, 98.17693947936652, 1320), (0.5758853356043497, 97.06937320611777, 1305), (0.5786626974741618, 96.79881779344703, 1290), (0.5816361983617147, 95.60365286588555, 1275), (0.5843882083892822, 94.36183482084346, 1260), (0.5873229344685872, 92.97664169647322, 1245), (0.5908117055892944, 91.8553163730652, 1230), (0.5939236720403035, 90.73985544085296, 1215), (0.5970321496327718, 87.15224397263725, 1200), (0.5998168230056763, 86.37705611161098, 1185), (0.6035075028737386, 85.53858776102305, 1170), (0.6070523500442505, 85.30424984479444, 1155), (0.6112357139587402, 84.93262595640405, 1140), (0.6183124701182048, 84.52429369261986, 1110), (0.6223223606745402, 84.33259475440582, 1095)]}, Backward:{0:[(1.074611775080363, 212.84139979987003, 1740), (1.0744989077250162, 213.39369998509653, 1725), (1.108141795794169, 212.65507911459227, 1425), (1.1122268199920655, 209.609588437588, 1410), (1.1160869598388672, 207.30109122471328, 1395), (1.120644458134969, 204.2275985068436, 1380), (1.1253358284632364, 201.5389420911136, 1365), (1.1298019488652546, 199.1393643104137, 1350), (1.1348331451416016, 197.06907142011804, 1335), (1.1396726687749228, 193.56264448069015, 1320), (1.144660266240438, 191.56078104337956, 1305), (1.1498423099517825, 189.06304998526105, 1290), (1.15516459941864, 187.14165830202154, 1275), (1.1603470166524252, 185.8983656494703, 1260), (1.1661136706670123, 183.7040151366105, 1245), (1.1717891057332357, 182.6081572726681, 1230), (1.1778138319651286, 179.57586651474975, 1215), (1.1839885234832763, 174.43412668775517, 1200), (1.1902702967325849, 172.01358904671537, 1185), (1.1969789028167723, 171.05626422535192, 1170), (1.2110523303349814, 169.33531087668342, 1140), (1.218172812461853, 168.17235560437302, 1125), (1.225728678703308, 167.8677670514102, 1110), (1.2417942921320595, 167.771199947639, 1080)], 1:[(1.0905425230662027, 214.05594546217065, 1740), (1.1251537879308064, 210.72033167392794, 1410), (1.1291558742523191, 207.56996672437413, 1395), (1.1334762175877888, 205.0817774765421, 1380), (1.1382328430811564, 202.97133952321911, 1365), (1.1427313645680746, 199.96476438210763, 1350), (1.1475585460662845, 197.11554261206845, 1335), (1.1526246865590413, 194.57938863843845, 1320), (1.157658759752909, 191.9466369692454, 1305), (1.1626705646514892, 189.74021797155723, 1290), (1.1679216066996256, 187.17251188547283, 1275), (1.1734012126922608, 184.8148152005757, 1260), (1.1845206181208292, 184.25240290226196, 1230), (1.190546409289042, 181.3672340816777, 1215), (1.1966176986694337, 180.35894149569668, 1200), (1.2029858668645224, 177.46666713515066, 1185), (1.2098310311635336, 174.14984268789266, 1170), (1.2167163928349811, 172.73244409052438, 1155), (1.2242037614186605, 170.52577254463176, 1140), (1.2311815818150835, 170.17735997005167, 1125), (1.239052653312683, 168.0787101986524, 1110), (1.2544997930526731, 167.82562116966068, 1080), (1.2803574323654174, 167.7371375151136, 1035), (1.299033522605896, 167.50659575563088, 1005)], 2:[(1.083737889925639, 209.61338044813226, 1740), (1.1088894844055175, 208.62136722617743, 1470), (1.1126289208730062, 207.04115113078018, 1455), (1.116676942507426, 203.63920821665423, 1440), (1.1207534551620484, 199.72761306255126, 1425), (1.1256544907887778, 196.92421677218312, 1410), (1.130063517888387, 194.45173767243085, 1395), (1.134632420539856, 193.57982884815033, 1380), (1.1391022284825645, 191.25641839178604, 1365), (1.1434833367665609, 188.92356889129513, 1350), (1.148462176322937, 186.44810826128867, 1335), (1.153561520576477, 184.83611916426605, 1320), (1.1577351729075114, 183.52691332134884, 1305), (1.1625194311141969, 181.00098325571886, 1290), (1.1688047568003337, 178.2438840902643, 1275), (1.1741213242212931, 175.6967524233017, 1260), (1.1797120730082191, 175.64891614338933, 1245), (1.1864797910054523, 174.6894859868241, 1230), (1.1909404198328657, 173.5783563706651, 1215), (1.1966481447219848, 169.18907625499668, 1200), (1.203143572807312, 167.09733742331323, 1185), (1.2174306631088256, 165.24252371852765, 1155), (1.2247342348098755, 163.90283879651776, 1140), (1.2309134642283122, 163.2634740964508, 1125), (1.254130736986796, 162.89037902288663, 1080), (1.262526567776998, 162.48217884865642, 1065)], 3:[(1.280308198928833, 251.42819708637654, 1740), (1.3249256531397502, 248.6240253169202, 1425), (1.3295952876408894, 244.77984373643883, 1410), (1.334486516316732, 241.8047578112658, 1395), (1.3400181849797568, 235.93843431910943, 1380), (1.346071990331014, 232.00189140701826, 1365), (1.3520028670628863, 229.62360124060382, 1350), (1.3581884066263834, 226.70078158208491, 1335), (1.364872924486796, 222.9982441459998, 1320), (1.3710870504379271, 222.97945985253253, 1305), (1.3779831647872924, 219.64496535280128, 1290), (1.3849554936091104, 216.54673129735409, 1275), (1.3919127702713012, 213.60573397466155, 1260), (1.3991417169570923, 210.6024392099631, 1245), (1.4070261398951212, 209.41218900277647, 1230), (1.4150474150975545, 206.75542430743508, 1215), (1.42335946559906, 198.1452366797477, 1200), (1.4316848119099934, 195.50375397113487, 1185), (1.4404025077819824, 194.47372997849266, 1170), (1.4499395767847696, 192.81020272007683, 1155), (1.4682388385136922, 192.8069379070551, 1125), (1.478088426589966, 192.1988996894888, 1110), (1.509665322303772, 191.6400219690613, 1065)]}}
)
# Schedule instructions with the "eager" scheduling algorithm.
# Refer to the docstring for available scheduling algorithms.
dag.schedule("pd")

# Pass the DAG to the pipeline visualizer.
# Refer to the constructor for matplotlib customization.
vis = PipelineVisualizer(dag)

# # Instantitate a matplotlib subplot and draw the pipeline and critical path.
fig, ax = plt.subplots(figsize=(12, 4), tight_layout=True)
vis.draw(ax, draw_time_axis=True)
vis.draw_critical_path(ax)
fig.savefig("pipeline.png")
