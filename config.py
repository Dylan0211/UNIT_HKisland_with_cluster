dis_learning_rate = 0.0001
gen_learning_rate = 0.001
num_epochs = 100
batch_size = 64
train_size = 0.5

temperature_min = 0
temperature_max = 40

"""
winter peak weekday,                0
winter average weekday,             1
winter average weekend day/holiday, 2

summer peak weekday,                3
summer average weekday,             4
summer average weekend day/holiday, 5

spring average weekday,             6 
fall average weekday                7
"""

# data prepare and train
n_clusters = 3
context_a = 1
context_b = 4

multiple_building_names = ['OIE', 'CP4', 'DEH']
target_building_name = 'LIH'


