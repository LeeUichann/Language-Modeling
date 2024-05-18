실험 환경
epoch = 100
optimizer = Adam, lr = 0.001
n_layers = 1
hidden_size = 128
batch_size = 512
length = 30
-RNN train_loss vs LSTM train_loss
![tt](https://github.com/LeeUichann/Language-Modeling/assets/166983272/e782ab2a-c292-465a-87be-c673b21faa8b) 

-RNN val_loss vs LSTM val_loss
![vv](https://github.com/LeeUichann/Language-Modeling/assets/166983272/322b8842-1c4b-428c-b40c-4cb9dc545122)

- val_loss를 비교하면 거의 비슷하지만 RNN이 미세하게 낮음


RNN-generate
 Temperature: 0.5

Sample 1:
In the dead of night and the the peat and the and the dear my math me the deand the sour the the pore the the sond the s
--------------------------------------------------

Sample 2:
A whisper in the wind and the beer he and the pore and the the the and the to the benous and what the to the the the bear
--------------------------------------------------

Sample 3:
The last light of day the of the his to the could the the me the the the sore will the the hear he the the seat the the s
--------------------------------------------------

Sample 4:
Through the looking glass the beat and the the sore the the the beare the will sere the here will the the hat the the the son
--------------------------------------------------

Sample 5:
Beyond the distant horizon the the sour the sore the the the the sour the seresers the preare the the say the the dot the hear
--------------------------------------------------

Temperature: 0.8

Sample 1:
In the dead of night,
Wads of thou the anous sell to the not the dot sore lome, wo the peacest und and with the Murdent 
--------------------------------------------------

Sample 2:
A whisper in the wind sont of concef
thar the of the ay het the non the denords, he so the mave dey ther me arat I the du
--------------------------------------------------

Sample 3:
The last light of day arat

CORUIUS:
Cot the hich the erart at hear if far thend, calle pore!

COLANUS:
Fi that and yould
--------------------------------------------------

Sample 4:
Through the looking glasse me oust mave thous say hat would of erent, his in all make tot cave the could thous of you theer m
--------------------------------------------------

Sample 5:
Beyond the distant horizond winst senderert he spead the pouse blave he ford not senom in what hak you not in the the my with 
--------------------------------------------------

Temperature: 1.2

Sample 1:
In the dead of nightong demty il
'siraN:: At, paym atfe!;
COLAMY?LOL: Aeatow hive
Hpeihe
Wne
reard araf YoE be Cifreguct
--------------------------------------------------

Sample 2:
A whisper in the wind;
Th, trer'sersviusitly, tamenowbit hes. He mearsw, fut my. gia! wise.
He usw!
MEo&th; io,
Tfoj:
Re 
--------------------------------------------------

Sample 3:
The last light of dayei peaPlcine.
L ing ale tul ou whhes bfe'm ma laogerust's mmyt aI bet's;:
Cot ines kout iply;
OMRhy 
--------------------------------------------------

Sample 4:
Through the looking glassed w'o? Fou
tulm a awaH:

Hamine?ect
Thut will bexu'cuat
.

FC;Cve.ble tromes-ing reas inf deal-
Aav
--------------------------------------------------


t값에 따른 결과 
t = 0.5, 반복적이고 문법적 오류가 발생하는 결과를 보임
t = 0.8, 좀 다양해지고 자연스러워짐
t = 1.2, 의미 없는 단어 조합 발생

해석
t값에 따라 확률분포의 모양이 달라지며 t = 0.8에서의 확률분포가 모델이 가장 높은 확률의 문자를 선택할 가능성을 높이면서도, 다양한 선택을 가능하게 하여 텍스트의 자연스러움을 유지한다고 볼 수 있음

미세하게 val_loss가 낮아 RNN을 선택했지만 LSTM에서 생성한 sample을 보면 t = 0.8일때 가장 자연스러웠음 

Temperature: 0.8 - LSTM

Sample 1:
In the dead of night anded
I ore the the ine you adrss we here.

Sore, the tide couther nof and destn to the mmand the m
--------------------------------------------------

Sample 2:
A whisper in the wind with you to nous wence thother hat the malled ungend all for your the sored the sour homtren camy t
--------------------------------------------------

Sample 3:
The last light of day the ther the heas' sutiter bust, you of fore the hisd the wot his bepe beces,
Serne well you be tar
--------------------------------------------------

Sample 4:
Through the looking glasser the do to the come.

Buve:
Mart dey, bus and th senter the now three maid the surond my well well
--------------------------------------------------

Sample 5:
Beyond the distant horizonn that sond foll ciny nof be wimt me tind with hast tow ard, to seet my menst wing.

