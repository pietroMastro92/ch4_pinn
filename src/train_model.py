import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import numpy as np
from sciann import Variable, Functional, SciModel, Parameter
from sciann.constraints import Data, MinMax
import sciann as sn 
import time

@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):
    """Function to train the model"""

    input_path = abspath(config.processed.path)
    output_path = abspath(config.final.path)

    print("Questi path saranno utili per tenere traccia dei vari set di parametri utilizzati per addestrare i modelli")
    print("il package hydra si occupa di tener traccia di tutte le configurazioni")
    print(f"Train modeling using {input_path}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {output_path}")



    # Synthetic data generated from sin function over [0, 2pi]
    x_true = np.linspace(0, np.pi*2, 10000)
    y_true = np.sin(x_true)
    dy_true = np.cos(x_true)

    # The network inputs should be defined with Variable.
    x = Variable('x')
    xf = Functional('xf', x)
    xf.set_trainable(False)

    # Each network is defined by Functional.
    y = Functional('y', x, [10, 10, 10], activation=['tanh', 'g-cos', 'l-sin'])
    dy_dx = sn.diff(y, x)

    d = Parameter(2.0, inputs=x)

    # Define the target (output) of your model.
    c1 = Data(y)

    # The model is formed with input `x` and condition `c1`.
    model = SciModel(x, [y, dy_dx], optimizer='adam')
    model.summary()

    start_time = time.time()

    # Training: .train runs the optimization and finds the parameters.
    model.train(x_true,
                [y_true, dy_true],
                epochs=10,
                learning_rate={"scheduler": "ExponentialDecay",
                            "initial_learning_rate": 1e-3,
                            "final_learning_rate": 1e-5,
                            "decay_epochs": 10,
                            "verify": False},
                batch_size=32,
                adaptive_weights={'method': "CLW", 'initial_weights': [0.1, 1.], 'final_weights': [2., 3.], 'curriculum_epochs': 20, "delay_epochs": 10},
                save_weights={'path': 'test', 'freq': 100}
                )

    print(f"Training finished in {time.time()-start_time}s. ")

    # used to evaluate the model after the training.
    y_pred = y.eval(model, x_true)

    # print(x_true.shape, y_pred.shape)
    import matplotlib.pyplot as plt
    plt.plot(x_true, y_true, '-k', x_true, y_pred, '--r')
    plt.show()


if __name__ == "__main__":
    train_model()
