from trainer import util
_, _, eval_x, eval_y = util.load_data()

prediction_input = eval_x.sample(5)
prediction_targets = eval_y[prediction_input.index]
print(prediction_input)
import json

with open('test.json', 'w') as json_file:
  for row in prediction_input.values.tolist():
    json.dump(row, json_file)
    json_file.write('\n')
