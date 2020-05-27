from numpy import array
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# split a univariate sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# Import sample dataset containing date, year, day of the week, day of the month, holidays and sales (i.e. #loans sold)
df = pd.read_csv('.\\data\\Sales.csv', delimiter=',', index_col='Date')

dates = df.index
# Split targets and features
Y = df.iloc[:, 4]
X = df.iloc[:, 0:4]

# define input sequence
raw_seq = [91735.78, 92522.67, 81501.16, 9964.79, 50736.35, 67692.23, 82539.61, 85880.87, 7176.26, 70918.97, 96102.8,
           56013.86, 60395.26, 84860.3, 82729.14, 14923.55, 62746.29, 64006.51, 39239.54, 41453.39, 70504.35, 77285.65,
           95980.75, 2242.63, 71640.55, 84646.81, 94148.42, 43154.31, 76231.98, 5002, 84648.68, 26926.31, 62336.47,
           44200.82, 51276.54, 41329.63, 49466.49, 66169.67, 57336.95, 9875.73, 59582.41, 85198.73, 9720.95, 77962.61,
           45553.45, 58826.64, 37590.98, 28722.66, 35615.73, 16125.05, 31448.02, 26161.53, 47774.94, 16010.45, 1737.09,
           85036.89, 81797.85, 39618.77, 74040.12, 48360.63, 19792.03, 51513.75, 70792.15, 94937.7, 25944.21, 7299.77,
           10541.81, 74320.04, 4264.42, 52685.46, 91388.94, 40624.47, 28763.47, 8568.46, 53918.85, 24706.56, 34434.78,
           43289.8, 93338.99, 97993.51, 6370.59, 39995, 17891.99, 56566.98, 34793.19, 67320.63, 1593.04, 6909.66,
           80376.42, 85904.29, 50452.24, 31777.94, 95920.09, 38915.75, 39531.02, 48359.56, 1361.3, 46332.31, 54358.82,
           14551.11, 41043.78, 69403.98, 41592.62, 20068.23, 30035.6, 84799.3, 48808.68, 49469.35, 5018.13, 3072.33,
           70188.11, 41884.77, 44811.91, 52907.64, 77258.92, 8395.85, 82884.48, 56689.28, 3562.51, 10822.84, 14223.48,
           76241.48, 76823.03, 14358.32, 63253.51, 83082.49, 4632.61, 96685.53, 39486.32, 38340.08, 35899.71, 14402.68,
           59538.65, 55784.19, 99495.1, 23052.9, 88564.56, 34190.47, 33080.19, 49651.33, 6033.27, 15825.2, 29982.14,
           36956.33, 90515.6, 66685.83, 37706.64, 54954.96, 89226.6, 27860.34, 19865.15, 98197.8, 37036.85, 45241.08,
           15515.42, 78901.27, 71728.05, 79394.03, 65882.41, 36563.65, 96015.27, 17969.15, 64693.49, 39907.18, 40142.43,
           48970.77, 36017.43, 36102.1, 34564.78, 56887.69, 72834.03, 91288.93, 90822.19, 81119.12, 46251.99, 12959.06,
           5451.12, 65061.27, 98604.74, 87234.39, 43127.6, 38711.6, 87198.83, 81372.81, 92440.82, 54395.89, 18381.21,
           29043.01, 39753.71, 96845.03, 85376.48, 41018.73, 41612.3, 52134.01, 57496.96, 5521.24, 12849.33, 70503.67,
           50475.96, 95192.56, 55125.81, 49054.95, 90007.04, 54114.97, 36197.88, 68095.14, 58794.62, 23419.17, 32733.1,
           59189.53, 28899.35, 13315.45, 87954.73, 88134.5, 89458.87, 49969.43, 63129.92, 86095.32, 33609.61, 73671.38,
           79165.26, 29006.39, 92126.77, 17024.87, 96198.19, 25526.27, 64034.33, 97993.08, 41677.31, 7484.16, 99740.03,
           76199.42, 50506.05, 14852.56, 54200.71, 8352.47, 48670.95, 55813.23, 47739.86, 44560.25, 50755.19, 79968.25,
           35906.76, 68261.51, 44766, 76443.56, 22734.74, 56656.8, 64358.47, 94525, 27070.58, 52420.09, 70937.85,
           55917.82, 42335.8, 56524.42, 74145.86, 84335.62, 29001.51, 60666.76, 47552.67, 72218.74, 1013.96, 23176.91,
           75734.97, 37451.62, 37984.97, 31632.28, 54472.18, 35585.31, 68050.09, 66285.74, 15314.3, 49496.94, 18075.5,
           82365.26, 55243.42, 52424.83, 97713.85, 82391.04, 48079.28, 27676.28, 10731.05, 92071.32, 57962.96, 69981.2,
           84860.63, 2059.91, 92895.31, 58376.99, 80659.81, 45745.34, 24495.01, 81798.24, 92289.46, 34330.28, 73546.14,
           72901.11, 17206.45, 86839.51, 85732.86, 42822.31, 6523.01, 14045.03, 5799.39, 50586.86, 60880.99, 76830.31,
           54146.16, 3422.04, 45863.45, 13789.52, 16525.61, 63972.82, 11055.68, 3125.03, 76776.05, 17837.39, 52723.67,
           56525.23, 17361.47, 80925.08, 51051.61, 90346.59, 35367.65, 84520.15, 66478.7, 42428.33, 97700.85, 93026.5,
           19214.97, 7541.57, 67792.8, 23241.73, 86622.87, 75478.27, 9308.67, 19805.94, 99802.34, 51427.82, 27460.71,
           35527.12, 13274.38, 64631.56, 66171.56, 61635.98, 34826.23, 77982.88, 20626.59, 32717.31, 51713.04, 47571.83,
           46630.94, 32466.47, 37560.47, 60143.44, 81043.18, 73012.78, 83752, 64307.03, 79169.53, 71246.57, 68894.31,
           56866.21, 80294.91, 55544.61, 97479.53, 64573.47, 95951.47, 67139.7, 52900.32, 45305.72, 1966.32, 2559.54,
           82140.55, 20166.12, 14968.09, 20020.98, 25593.78, 47988.88, 68961.54, 37781.9, 96224.64, 6915.06, 25222.59,
           70088.17, 52212.04, 96618.9, 14394.86, 16727.78, 46970.07, 69143.03, 34247.79, 67773.92, 55832.31, 98796.57,
           37634.9, 89747.92, 70971.61, 83881.59, 48260.54, 52484.36, 27666.48, 26596.64, 25476.92, 28815.35, 26114.11,
           64501.82, 50323.92, 59369, 64368.61, 75031.32, 73191.97, 42678.73, 48313.91, 14405.33, 95108.46, 10135.42,
           29966.89, 25527.55, 44419.21, 72467.93, 83592.74, 40647.49, 19352.94, 78809.3, 16065.21, 48302.79, 52806.34,
           93251.49, 56661.25, 83916.05, 79160.12, 24030.55, 44551.99, 69473.55, 72114.38, 68676, 28339.77, 19123.74,
           34864.34, 13692.14, 25151.44, 36910.72, 5800.88, 79552.49, 7857.84, 46256.9, 26905.46, 7298.11, 38998.26,
           4404.48, 15640.89, 62354.55, 74945.03, 22082.59, 42426.3, 4261.23, 95874, 99387.43, 7569.53, 2801.62,
           65170.63, 5874.7, 84761.99, 39956.27, 27158.23, 16150.94, 84806.18, 73598.42, 11880.04, 9500.87, 32000.27,
           2409.8, 16297.93, 37977.36, 16759.03, 57474.99, 13702.36, 86995.72, 64035.63, 36841.29, 96534.32, 94103.45,
           84049.55, 2778.53, 87319.07, 13419.18, 59031.71, 34735.02, 40461.69, 897.56, 28356.61, 684.25, 75737.38,
           52261.39, 47034.04, 63714.99, 87987.96, 37356.54, 80427.27, 34721.31, 19574.59, 38451.52, 28570.52, 48817.85,
           3552.27, 14418.95, 54407.62, 50596.31, 11079.75, 89851.22, 35784.21, 60711.33, 89930.69, 76288.25, 97065.08,
           55708.68, 57720.38, 47492.01, 58537.67, 75165.69, 67804.77, 30469.16, 45250.59, 3713.14, 25729.05, 85480.05,
           95973.35, 65957.56, 76913.33, 22797.01, 84299.78, 50179.19, 58740.38, 9660.45, 32229.32, 19622.61, 7467.22,
           31704.38, 7298.35, 62766.3, 17815.65, 36532.61, 20335.05, 49344.16, 25198.61, 15769.93, 51829.66, 73901.5,
           23129.35, 91116.22, 18577.4, 27169.26, 96730, 42552.63, 28784.3, 83933.38, 92879.15, 28083.25, 58281.25,
           99849.05, 92627.29, 25783.28, 63637.92, 59504.22, 43288.59, 79615.86, 60615.04, 44005.23, 43368.08, 20197.53,
           79315.58, 47854.45, 96142.54, 89127.59, 92345.88, 38704.82, 54431.36, 47171.78, 45148.09, 41189.53, 67562.02,
           79586.17, 61357.19, 34871.08, 11191.58, 88077.33, 35679.24, 27011.11, 14770.97, 5363.89, 69206.94, 23594.41,
           11660.77, 70777.67, 47148.95, 7708.41, 58748.22, 10229.99, 57812.51, 12297.88, 40466.81, 50740.53, 54755.18,
           25363.99, 94069.14, 97885.99, 45266.9, 72157.8, 53545.54, 13629.63, 13855.6, 9951.53, 58169.23, 58935.49,
           10734.19, 90852.95, 49616.61, 69678.3, 41925.87, 52254.46, 47445.2, 47992.11, 24550.49, 61549.09, 68072.23,
           33189.61, 19300.71, 92243.67, 32112.53, 70436.07, 68999.24, 86495.61, 51350.27, 74649.64, 11409.94, 94339.37,
           48695.7, 11572.33, 2276.19, 72636.24, 31995.94, 88653.83, 41262.94, 24653.83, 19224.24, 29643.17, 24089.38,
           79727.43, 42682.76, 50369.42, 61216.62, 59872.64, 17180.24, 36473.02, 39214.21, 34745.01, 48490.8, 70176.84,
           19215.41, 98178.83, 68311.79, 25797.2, 60697.15, 43666.19, 51984.44, 92294.87, 59132.41, 65569.65, 83843.03,
           34260.79, 65497.9, 10060.1, 56987.04, 78410.93, 97536.39, 10328.12, 58134.66, 26464.37, 99353.77, 47576.85,
           26680.52, 71472.06, 75268.5, 14647.94, 75411.4, 31361.56, 62401.81, 12251.89, 45022.27, 25332.39, 52734.93,
           30353.59, 32210.04, 92786.93, 28132.59, 31980.15, 69356.09, 62063.17, 79677.57, 79011.65, 9677.11, 44219.89,
           67634.16, 69659.8, 77891.71, 89254.16, 58394.78, 24000.23, 3239.31, 24229.62, 35510.48, 17712.76, 55557.49,
           91122.85, 70900.32, 67764.38, 96448.43, 43105.29, 91186.73, 41723.64, 19763.53, 63097.74, 17943.71, 70561.97,
           64240.98, 63073.24, 97553.12, 93995.23, 61490.01, 39939.62, 50420.15, 28349.61, 66167.17, 48236.59, 82916.26,
           99361.83, 26573.53, 22428.17, 75929.68, 74692.35, 17474.22, 34459.34, 47560.87, 1944.6, 45455.61, 71183.14,
           56712.87, 8952.13, 656.69, 3498.11, 86119.75, 63086.37, 82451.8, 49974.56, 62924.42, 90124.61, 81000.14,
           75106.33, 30540.88, 45206.91, 95244.68, 8390.18, 17472.32, 81956.08, 58167.37, 81094.63, 64361.03, 77140.74,
           21779.11, 70819.76, 54256.03, 39450.64, 4343.26, 22452.73, 884.22, 1306.69, 34500.75, 55346.6, 20346.2,
           13361, 81158.06, 8054.76, 68468.28, 94487.07, 70470.99, 68634.7, 27239.63, 57685.17, 43399.87, 87737.08,
           2035.65, 38967.22, 7095.46, 7716.07, 41189.95, 63183.5, 60621.72, 42838.69, 72575.72, 58493.98, 89506.4,
           35036.8, 95971.48, 68995.9, 51215.09, 42538.85, 38888.78, 77843.27, 35209.59, 43509.85, 88524.27, 76341.12,
           95282.65, 76454.02, 24622.27, 20247.95, 60606.23, 74743.32, 78636.82, 47296.8, 88350.18, 39956.52, 5541.24,
           85276.02, 48630.45, 81734.25, 43538.63, 70853.73, 76447.68, 27466.44, 78906.8, 96865.76, 63426.21, 64610.38,
           21425.18, 63134.46, 97912.06, 59040.57, 74826.95, 38237.89, 68894.13, 44017.24, 3873.5, 99099.19, 89918.92,
           89938.49, 52907.8, 98680.98, 65821.86, 95026.81, 90959.19, 34964.99, 71308.05, 99209.63, 31127.43, 86022.24,
           2786.92, 65457.3, 57119.95, 40776.61, 16398.4, 45911.69, 99314.02, 58257.5, 71449.97, 53405, 4279.58,
           80084.75, 44233, 15336.18, 99681.23, 79215.24, 68244.39, 57032.84, 68670.05, 3807.79, 21486.54, 6410.06,
           82332.61, 87391.92, 65358.02, 7128.43, 96162.07, 62525.2, 25477.34, 68813.75, 84689.3, 99793.9, 88478.18,
           44477.9, 28884.19, 79661.95, 47844.41, 66244.81, 93814.28, 28894.09, 22778.73, 51370.54, 56768.66, 45992.3,
           49894.95, 91073.45, 90006.59, 31563.67, 35250.28, 25548.29, 81675.05, 62436.3, 19843.88, 65726.26, 44115.76,
           47994.77, 23606.81, 98773.63, 87791.44, 31847.99, 89083.61, 65684.97, 32126.59, 12100.42, 88463.19, 88205.58,
           31401.73, 48735.39, 86321.14, 7513.44, 72874.51, 7859.76, 25291.39, 24496.3, 58621.2, 80973.21, 98037.1,
           9895.07, 30828.21, 55639.49, 76741.78, 69593.57, 82132.25, 50006.71, 50136.39, 85731.29, 83584.83, 49374.68,
           51313.81, 25601.82, 70577.35, 22236.81, 33958.15, 7468.1, 78867.91, 10213.28, 91114.44, 60898.14, 79100.62,
           11408.18, 87264.94, 13996.74, 170.04, 24065.69, 63759.11, 57012.69, 87402.74, 64760.91, 9297.77, 63583.57,
           64935.24, 80111.9, 53838.1, 21139.68, 87422.88, 17597.97, 78260.45, 20879.99, 56755, 39276.47, 86921.46,
           59546.83, 58778.56, 57197.96, 71426.3, 83796.09, 13351.68, 75690.28, 52446.99, 28221.03, 49063.29, 8752.53,
           42229.72, 93845.72, 54468.75, 15071.77, 59742.83, 64564.67, 84668.77, 11723.26, 84609.18, 66368.51
           ]
# choose a number of time steps
n_steps = 10
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([9223,
                 6173,
                 2277,
                 8719,
                 1618,
                 9807,
                 7765,
                 636,
                 6040,
                 250]
                )
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
