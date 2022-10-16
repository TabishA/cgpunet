from tensorflow import keras
from keras import backend as K
from keras_gcn import GraphConv
import numpy as np
from tqdm import tqdm, trange
from simgnn.src.parser import parameter_parser
from simgnn.src.utilities import data2, convert_to_keras, process, find_loss
from simgnn.src.simgnn import simgnn
from simgnn.src.custom_layers import CustomAttention, NeuralTensorLayer
from keras.backend import manual_variable_initialization
import pickle
manual_variable_initialization(True)

parser = parameter_parser()

def train(model, x):
    batches = x.create_batches(batch_size=16)
    global_labels = x.getlabels()
    """
    Training the Network
    Take every graph pair and train it as a batch.
    """
    t_x = x
    last=0
    for epoch in range(0,parser.epochs):
        p=0
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for graph_pair in batch:
                data = process(graph_pair)
                data = convert_to_keras(data, global_labels)
                x, y, a, b = [ np.array([ data["features_1"] ]), np.array([ data["features_2"] ]), np.array([ data["edge_index_1"] ]), np.array([ data["edge_index_2"] ]) ]
                p = model.train_on_batch([x, a, y, b], data["target"])
        if epoch%(parser.saveafter) == 0:
                print("Train Error:")
                print(p)
                model.save("train")
                model.save_weights("xweights")

def test(model, x):
    global_labels = x.getlabels()
    test = x.gettest()
    scores = []
    g_truth = []
    for graph_pair in tqdm(test):
        data = process(graph_pair)
        data = convert_to_keras(data, global_labels)
        x = np.array([ data["features_1"] ])
        y = np.array([ data["features_2"] ])
        a = np.array([ data["edge_index_1"] ])
        b = np.array([ data["edge_index_2"] ])
        g_truth.append(data["target"])
        y=model.predict([x, a, y, b])
        scores.append(find_loss(y, data["target"]))

    norm_ged_mean = np.mean(g_truth)
    model_error = np.mean(scores)
    print("\nModel test error: " +str(round(model_error, 5))+".")
    return model_error

def main():
    model = simgnn(parser);
    opt = keras.optimizers.Adadelta(learning_rate=parser.learning_rate, rho=parser.weight_decay)
    #opt = keras.optimizers.Adam(learning_rate=parser.learning_rate)
    model.compile(
                optimizer=opt,
                loss='mse',
                metrics=[keras.metrics.MeanSquaredError()],
            )
    model.summary()
    model.save("simgnn")
    """"
    x : Data loading
    train used to train
    test over the test data
    """
    x = data2()
    pickle.dump(x.global_labels, open("./global_labels.p", 'wb'))
    train(model, x)
    test(model, x)


if __name__ == "__main__":
    # main()

    data = process('../dataset/test/83.json')
    # global_labels = {'6': 0, '1': 1, '0': 2, '3': 3, '5': 4, '2': 5, '4': 6, '7': 7}
    global_labels = pickle.load(open('./global_labels.p', 'rb'))
    scaling_factor = 0.5 * (len(data["labels_1"]) + len(data["labels_2"]))
    ged = data['ged']
    data = convert_to_keras(data, global_labels)

    x = np.array([data["features_1"]])
    y = np.array([data["features_2"]])
    a = np.array([data["edge_index_1"]])
    b = np.array([data["edge_index_2"]])

    model = keras.models.load_model("train")
    pred_log = model.predict([x, a, y, b])
    pred_log = pred_log[0][0]
    pred_norm = -np.log(pred_log)/np.log(np.exp(1))
    pred = pred_norm*scaling_factor

    print("prediction: {}".format(pred))
    print("true ged: {}".format(ged))


