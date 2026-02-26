import os
import pandas as pd
import io

# =========================================================
# [1] 설정
# =========================================================
WORK_DIR = r'C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY7-2\images'
EQUIPMENT_NAME = "E3S690G3"

# =========================================================
# [2] 데이터 입력
# =========================================================
# 엑셀 데이터를 여기에 복사해서 붙여넣으세요.
RAW_DATA = """
Round 1	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	1	X	1	1		134973
	2	D	1	Knives 	2	5	Knitting 	2	X	2	1		134974
	3	D	2	Razor	1	5	Knitting 	3	X	3	1		134975
	4	D	2	Razor	2	6	Printer Cartridge	1	X	4	1		134976
	5	D	3	Matchbox	1	6	Printer Cartridge	2	X	5	1		134977
	6	D	3	Matchbox	2	6	Printer Cartridge	3	X	6	1		134978
	7	D	4	Wrenches	1	7	Scissors	1	X	7	1		134979
	8	D	4	Wrenches	2	7	Scissors	2	X	8	1		134980
	9	D	4	Wrenches	3	7	Scissors	3	X	9	1		134981
	10	D	4	Wrenches	4	7	Scissors	4	X	10	1		134982
	11	D	4	Wrenches	5	5	Knitting 	1	X	1	1		134983
Round 2	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	2	X	1	1		134984
	2	D	1	Knives 	2	5	Knitting 	3	X	2	1		134985
	3	D	2	Razor	1	6	Printer Cartridge	1	X	3	1		134986
	4	D	2	Razor	2	6	Printer Cartridge	2	X	4	1		134987
	5	D	3	Matchbox	1	6	Printer Cartridge	3	X	5	1		134988
	6	D	3	Matchbox	2	7	Scissors	1	X	6	1		134989
	7	D	4	Wrenches	1	7	Scissors	2	X	7	1		134990
	8	D	4	Wrenches	2	7	Scissors	3	X	8	1		134991
	9	D	4	Wrenches	3	7	Scissors	4	X	9	1		134992
	10	D	4	Wrenches	4	5	Knitting 	1	X	10	1		134993
	11	D	4	Wrenches	5	5	Knitting 	2	X	1	1		134994
Round 3	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	3	X	1	1		134995
	2	D	1	Knives 	2	6	Printer Cartridge	1	X	2	1		134996
	3	D	2	Razor	1	6	Printer Cartridge	2	X	3	1		134997
	4	D	2	Razor	2	6	Printer Cartridge	3	X	4	1		134998
	5	D	3	Matchbox	1	7	Scissors	1	X	5	1		134999
	6	D	3	Matchbox	2	7	Scissors	2	X	6	1		135000
	7	D	4	Wrenches	1	7	Scissors	3	X	7	1		135001
	8	D	4	Wrenches	2	7	Scissors	4	X	8	1		135002
	9	D	4	Wrenches	3	5	Knitting 	1	X	9	1		135003
	10	D	4	Wrenches	4	5	Knitting 	2	X	10	1		135004
	11	D	4	Wrenches	5	5	Knitting 	3	X	1	1		135005
Round 4	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	1	X	1	1		135006
	2	D	1	Knives 	2	6	Printer Cartridge	2	X	2	1		135007
	3	D	2	Razor	1	6	Printer Cartridge	3	X	3	1		135008
	4	D	2	Razor	2	7	Scissors	1	X	4	1		135009
	5	D	3	Matchbox	1	7	Scissors	2	X	5	1		135010
	6	D	3	Matchbox	2	7	Scissors	3	X	6	1		135011
	7	D	4	Wrenches	1	7	Scissors	4	X	7	1		135012
	8	D	4	Wrenches	2	5	Knitting 	1	X	8	1		135013
	9	D	4	Wrenches	3	5	Knitting 	2	X	9	1		135014
	10	D	4	Wrenches	4	5	Knitting 	3	X	10	1		135015
	11	D	4	Wrenches	5	6	Printer Cartridge	1	X	1	1		135016
Round 5	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	2	X	1	1		135017
	2	D	1	Knives 	2	6	Printer Cartridge	3	X	2	1		135018
	3	D	2	Razor	1	7	Scissors	1	X	3	1		135019
	4	D	2	Razor	2	7	Scissors	2	X	4	1		135020
	5	D	3	Matchbox	1	7	Scissors	3	X	5	1		135021
	6	D	3	Matchbox	2	7	Scissors	4	X	6	1		135022
	7	D	4	Wrenches	1	5	Knitting 	1	X	7	1		135023
	8	D	4	Wrenches	2	5	Knitting 	2	X	8	1		135024
	9	D	4	Wrenches	3	5	Knitting 	3	X	9	1		135025
	10	D	4	Wrenches	4	6	Printer Cartridge	1	X	10	1		135026
	11	D	4	Wrenches	5	6	Printer Cartridge	2	X	1	1		135027
Round 6	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	3	X	1	1		135028
	2	D	1	Knives 	2	7	Scissors	1	X	2	1		135029
	3	D	2	Razor	1	7	Scissors	2	X	3	1		135030
	4	D	2	Razor	2	7	Scissors	3	X	4	1		135031
	5	D	3	Matchbox	1	7	Scissors	4	X	5	1		135032
	6	D	3	Matchbox	2	5	Knitting 	1	X	6	1		135033
	7	D	4	Wrenches	1	5	Knitting 	2	X	7	1		135034
	8	D	4	Wrenches	2	5	Knitting 	3	X	8	1		135035
	9	D	4	Wrenches	3	6	Printer Cartridge	1	X	9	1		135036
	10	D	4	Wrenches	4	6	Printer Cartridge	2	X	10	1		135037
	11	D	4	Wrenches	5	6	Printer Cartridge	3	X	1	1		135038
Round 7	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	1	X	1	1		135039
	2	D	1	Knives 	2	7	Scissors	2	X	2	1		135040
	3	D	2	Razor	1	7	Scissors	3	X	3	1		135041
	4	D	2	Razor	2	7	Scissors	4	X	4	1		135042
	5	D	3	Matchbox	1	5	Knitting 	1	X	5	1		135043
	6	D	3	Matchbox	2	5	Knitting 	2	X	6	1		135044
	7	D	4	Wrenches	1	5	Knitting 	3	X	7	1		135045
	8	D	4	Wrenches	2	6	Printer Cartridge	1	X	8	1		135046
	9	D	4	Wrenches	3	6	Printer Cartridge	2	X	9	1		135047
	10	D	4	Wrenches	4	6	Printer Cartridge	3	X	10	1		135048
	11	D	4	Wrenches	5	7	Scissors	1	X	1	1		135049
Round 8	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	2	X	1	1		135050
	2	D	1	Knives 	2	7	Scissors	3	X	2	1		135051
	3	D	2	Razor	1	7	Scissors	4	X	3	1		135052
	4	D	2	Razor	2	5	Knitting 	1	X	4	1		135053
	5	D	3	Matchbox	1	5	Knitting 	2	X	5	1		135054
	6	D	3	Matchbox	2	5	Knitting 	3	X	6	1		135055
	7	D	4	Wrenches	1	6	Printer Cartridge	1	X	7	1		135056
	8	D	4	Wrenches	2	6	Printer Cartridge	2	X	8	1		135057
	9	D	4	Wrenches	3	6	Printer Cartridge	3	X	9	1		135059
	10	D	4	Wrenches	4	7	Scissors	1	X	10	1		135060
	11	D	4	Wrenches	5	7	Scissors	2	X	1	1		135061
Round 9	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	3	X	1	1		135062
	2	D	1	Knives 	2	7	Scissors	4	X	2	1		135063
	3	D	2	Razor	1	5	Knitting 	1	X	3	1		135064
	4	D	2	Razor	2	5	Knitting 	2	X	4	1		135065
	5	D	3	Matchbox	1	5	Knitting 	3	X	5	1		135066
	6	D	3	Matchbox	2	6	Printer Cartridge	1	X	6	1		135068
	7	D	4	Wrenches	1	6	Printer Cartridge	2	X	7	1		135069
	8	D	4	Wrenches	2	6	Printer Cartridge	3	X	8	1		135070
	9	D	4	Wrenches	3	7	Scissors	1	X	9	1		135071
	10	D	4	Wrenches	4	7	Scissors	2	X	10	1		135072
	11	D	4	Wrenches	5	7	Scissors	3	X	1	1		135073
Round 10	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	4	X	1	1		135075
	2	D	1	Knives 	2	5	Knitting 	1	X	2	1		135076
	3	D	2	Razor	1	5	Knitting 	2	X	3	1		135077
	4	D	2	Razor	2	5	Knitting 	3	X	4	1		135078
	5	D	3	Matchbox	1	6	Printer Cartridge	1	X	5	1		135079
	6	D	3	Matchbox	2	6	Printer Cartridge	2	X	6	1		135080
	7	D	4	Wrenches	1	6	Printer Cartridge	3	X	7	1		135081
	8	D	4	Wrenches	2	7	Scissors	1	X	8	1		135082
	9	D	4	Wrenches	3	7	Scissors	2	X	9	1		135083
	10	D	4	Wrenches	4	7	Scissors	3	X	10	1		135084
	11	D	4	Wrenches	5	7	Scissors	4	X	1	1		135085
Round 11	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	1	X	1	2		135181
	2	D	1	Knives 	2	5	Knitting 	2	X	2	2		135182
	3	D	2	Razor	1	5	Knitting 	3	X	3	2		135183
	4	D	2	Razor	2	6	Printer Cartridge	1	X	4	2		135184
	5	D	3	Matchbox	1	6	Printer Cartridge	2	X	5	2		135185
	6	D	3	Matchbox	2	6	Printer Cartridge	3	X	6	2		135186
	7	D	4	Wrenches	1	7	Scissors	1	X	7	2		135187
	8	D	4	Wrenches	2	7	Scissors	2	X	8	2		135188
	9	D	4	Wrenches	3	7	Scissors	3	X	9	2		135189
	10	D	4	Wrenches	4	7	Scissors	4	X	10	2		135190
	11	D	4	Wrenches	5	5	Knitting 	1	X	1	2		135191
Round 12	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	2	X	1	2		135192
	2	D	1	Knives 	2	5	Knitting 	3	X	2	2		135193
	3	D	2	Razor	1	6	Printer Cartridge	1	X	3	2		135194
	4	D	2	Razor	2	6	Printer Cartridge	2	X	4	2		135195
	5	D	3	Matchbox	1	6	Printer Cartridge	3	X	5	2		135196
	6	D	3	Matchbox	2	7	Scissors	1	X	6	2		135197
	7	D	4	Wrenches	1	7	Scissors	2	X	7	2		135198
	8	D	4	Wrenches	2	7	Scissors	3	X	8	2		135199
	9	D	4	Wrenches	3	7	Scissors	4	X	9	2		135200
	10	D	4	Wrenches	4	5	Knitting 	1	X	10	2		135201
	11	D	4	Wrenches	5	5	Knitting 	2	X	1	2		135203
Round 13	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	5	Knitting 	3	X	1	2		135204
	2	D	1	Knives 	2	6	Printer Cartridge	1	X	2	2		135205
	3	D	2	Razor	1	6	Printer Cartridge	2	X	3	2		135206
	4	D	2	Razor	2	6	Printer Cartridge	3	X	4	2		135207
	5	D	3	Matchbox	1	7	Scissors	1	X	5	2		135209
	6	D	3	Matchbox	2	7	Scissors	2	X	6	2		135210
	7	D	4	Wrenches	1	7	Scissors	3	X	7	2		135211
	8	D	4	Wrenches	2	7	Scissors	4	X	8	2		135212
	9	D	4	Wrenches	3	5	Knitting 	1	X	9	2		135213
	10	D	4	Wrenches	4	5	Knitting 	2	X	10	2		135215
	11	D	4	Wrenches	5	5	Knitting 	3	X	1	2		135216
Round 14	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	1	X	1	2		135217
	2	D	1	Knives 	2	6	Printer Cartridge	2	X	2	2		135218
	3	D	2	Razor	1	6	Printer Cartridge	3	X	3	2		135219
	4	D	2	Razor	2	7	Scissors	1	X	4	2		135220
	5	D	3	Matchbox	1	7	Scissors	2	X	5	2		135221
	6	D	3	Matchbox	2	7	Scissors	3	X	6	2		135222
	7	D	4	Wrenches	1	7	Scissors	4	X	7	2		135223
	8	D	4	Wrenches	2	5	Knitting 	1	X	8	2		135224
	9	D	4	Wrenches	3	5	Knitting 	2	X	9	2		135225
	10	D	4	Wrenches	4	5	Knitting 	3	X	10	2		135226
	11	D	4	Wrenches	5	6	Printer Cartridge	1	X	1	2		135227
Round 15	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	2	X	1	2		135228
	2	D	1	Knives 	2	6	Printer Cartridge	3	X	2	2		135229
	3	D	2	Razor	1	7	Scissors	1	X	3	2		135230
	4	D	2	Razor	2	7	Scissors	2	X	4	2		135231
	5	D	3	Matchbox	1	7	Scissors	3	X	5	2		135232
	6	D	3	Matchbox	2	7	Scissors	4	X	6	2		135233
	7	D	4	Wrenches	1	5	Knitting 	1	X	7	2		135234
	8	D	4	Wrenches	2	5	Knitting 	2	X	8	2		135235
	9	D	4	Wrenches	3	5	Knitting 	3	X	9	2		135236
	10	D	4	Wrenches	4	6	Printer Cartridge	1	X	10	2		135237
	11	D	4	Wrenches	5	6	Printer Cartridge	2	X	1	2		135238
Round 16	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	6	Printer Cartridge	3	X	1	2		135239
	2	D	1	Knives 	2	7	Scissors	1	X	2	2		135240
	3	D	2	Razor	1	7	Scissors	2	X	3	2		135241
	4	D	2	Razor	2	7	Scissors	3	X	4	2		135242
	5	D	3	Matchbox	1	7	Scissors	4	X	5	2		135243
	6	D	3	Matchbox	2	5	Knitting 	1	X	6	2		135244
	7	D	4	Wrenches	1	5	Knitting 	2	X	7	2		135245
	8	D	4	Wrenches	2	5	Knitting 	3	X	8	2		135246
	9	D	4	Wrenches	3	6	Printer Cartridge	1	X	9	2		135247
	10	D	4	Wrenches	4	6	Printer Cartridge	2	X	10	2		135248
	11	D	4	Wrenches	5	6	Printer Cartridge	3	X	1	2		135249
Round 17	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	1	X	1	2		135250
	2	D	1	Knives 	2	7	Scissors	2	X	2	2		135251
	3	D	2	Razor	1	7	Scissors	3	X	3	2		135252
	4	D	2	Razor	2	7	Scissors	4	X	4	2		135253
	5	D	3	Matchbox	1	5	Knitting 	1	X	5	2		135254
	6	D	3	Matchbox	2	5	Knitting 	2	X	6	2		135255
	7	D	4	Wrenches	1	5	Knitting 	3	X	7	2		135256
	8	D	4	Wrenches	2	6	Printer Cartridge	1	X	8	2		135257
	9	D	4	Wrenches	3	6	Printer Cartridge	2	X	9	2		135258
	10	D	4	Wrenches	4	6	Printer Cartridge	3	X	10	2		135259
	11	D	4	Wrenches	5	7	Scissors	1	X	1	2		135260
Round 18	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	2	X	1	2		135261
	2	D	1	Knives 	2	7	Scissors	3	X	2	2		135262
	3	D	2	Razor	1	7	Scissors	4	X	3	2		135263
	4	D	2	Razor	2	5	Knitting 	1	X	4	2		135264
	5	D	3	Matchbox	1	5	Knitting 	2	X	5	2		135265
	6	D	3	Matchbox	2	5	Knitting 	3	X	6	2		135266
	7	D	4	Wrenches	1	6	Printer Cartridge	1	X	7	2		135267
	8	D	4	Wrenches	2	6	Printer Cartridge	2	X	8	2		135268
	9	D	4	Wrenches	3	6	Printer Cartridge	3	X	9	2		135269
	10	D	4	Wrenches	4	7	Scissors	1	X	10	2		135270
	11	D	4	Wrenches	5	7	Scissors	2	X	1	2		135271
Round 19	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	3	X	1	2		135272
	2	D	1	Knives 	2	7	Scissors	4	X	2	2		135273
	3	D	2	Razor	1	5	Knitting 	1	X	3	2		135274
	4	D	2	Razor	2	5	Knitting 	2	X	4	2		135275
	5	D	3	Matchbox	1	5	Knitting 	3	X	5	2		135276
	6	D	3	Matchbox	2	6	Printer Cartridge	1	X	6	2		135277
	7	D	4	Wrenches	1	6	Printer Cartridge	2	X	7	2		135278
	8	D	4	Wrenches	2	6	Printer Cartridge	3	X	8	2		135279
	9	D	4	Wrenches	3	7	Scissors	1	X	9	2		135280
	10	D	4	Wrenches	4	7	Scissors	2	X	10	2		135281
	11	D	4	Wrenches	5	7	Scissors	3	X	1	2		135282
Round 20	No	Scenario	Threat ID 1	Threat Item 1	Sample ID 1	Threat ID 2	Threat Item 2	Sample ID 2	Folding	Bag ID	Background Item Type	PhotoID	UID
	1	D	1	Knives 	1	7	Scissors	4	X	1	2		135283
	2	D	1	Knives 	2	5	Knitting 	1	X	2	2		135284
	3	D	2	Razor	1	5	Knitting 	2	X	3	2		135285
	4	D	2	Razor	2	5	Knitting 	3	X	4	2		135286
	5	D	3	Matchbox	1	6	Printer Cartridge	1	X	5	2		135287
	6	D	3	Matchbox	2	6	Printer Cartridge	2	X	6	2		135288
	7	D	4	Wrenches	1	6	Printer Cartridge	3	X	7	2		135289
	8	D	4	Wrenches	2	7	Scissors	1	X	8	2		135290
	9	D	4	Wrenches	3	7	Scissors	2	X	9	2		135291
	10	D	4	Wrenches	4	7	Scissors	3	X	10	2		135292
	11	D	4	Wrenches	5	7	Scissors	4	X	1	2		135293
"""

# =========================================================
# [3] 내부 로직
# =========================================================
def get_code_from_value(column_name, value):
    val_str = str(value).strip()
    
    # Position, Orientation 관련 로직 삭제됨
    if column_name == 'Folding':
        if val_str == 'X': return "0"
        elif val_str == 'O': return "1"
        return "0"
    elif column_name == 'Background Item Type': 
        return val_str
    
    return val_str

def run_renaming():
    print("데이터 분석 중...")
    try:
        if not RAW_DATA.strip():
            print("[오류] RAW_DATA가 비어있습니다. [2] 데이터 입력란에 엑셀 내용을 붙여넣어주세요.")
            return

        df = pd.read_csv(io.StringIO(RAW_DATA), sep='\t')
        df = df[df['UID'] != 'UID'] 
        df = df.dropna(subset=['UID'])
        
        # UID 8자리 맞추기 (00 채우기)
        df['UID'] = df['UID'].astype(str).replace(r'\.0$', '', regex=True).str.strip()
        df['UID'] = df['UID'].apply(lambda x: x.zfill(8))
        
    except Exception as e:
        print(f"[오류] 데이터 로드 실패: {e}")
        return

    files = os.listdir(WORK_DIR)
    count = 0
    print(f"\n총 {len(df)}개 데이터 로드됨 (UID 00 채움 완료). 변경 시작...\n")

    for filename in files:
        name, ext = os.path.splitext(filename)
        if name.startswith('.'): continue

        # 파일명 처리를 시작하기 전에 '_PH' 제거
        if name.endswith('_PH'):
            name = name[:-3] 

        try:
            if '_' in name:
                file_uid = name.split('_')[0] 
                view_no = name.split('_')[-1]
            else:
                file_uid = name
                view_no = "1"
            
            # 엑셀 데이터 매칭
            row = df[df['UID'] == file_uid]
            
            if not row.empty:
                data = row.iloc[0]
                
                scenario = str(data['Scenario']).strip()
                
                # Threat 1 정보 추출
                threat_id_1 = str(data['Threat ID 1']).strip()
                threat_item_1 = str(data['Threat Item 1']).strip().replace(" ", "-")
                sample_id_1 = str(data['Sample ID 1']).strip()
                
                # Threat 2 정보 추출
                threat_id_2 = str(data['Threat ID 2']).strip()
                threat_item_2 = str(data['Threat Item 2']).strip().replace(" ", "-")
                sample_id_2 = str(data['Sample ID 2']).strip()
                
                folding = get_code_from_value('Folding', data['Folding'])
                bag_id = str(data['Bag ID']).strip()
                bg = get_code_from_value('Background Item Type', data['Background Item Type'])
                
                # 변경된 네이밍 규칙 적용
                new_name = (
                    f"{EQUIPMENT_NAME}_{file_uid}_{scenario}_"
                    f"{threat_id_1}_{threat_item_1}_{sample_id_1}_"
                    f"{threat_id_2}_{threat_item_2}_{sample_id_2}_"
                    f"{folding}_{bag_id}_{bg}_{view_no}{ext}"
                )
                
                old_path = os.path.join(WORK_DIR, filename)
                new_path = os.path.join(WORK_DIR, new_name)
                
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"[변경] {filename} -> {new_name}")
                    count += 1
            else:
                pass
                
        except Exception as e:
            print(f"[에러] {filename}: {e}")

    print(f"\n작업 종료: 총 {count}개 변경됨")

if __name__ == "__main__":
    run_renaming()