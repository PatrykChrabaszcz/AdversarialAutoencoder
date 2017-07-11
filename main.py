import tensorflow as tf
from src.datasets import Celeb
from src.model_res import Model
from src.model_classic import ModelClassic
from src.estimator import Estimator
import time
import click

dataset_path = "/home/chrabasp/Download/img_align_celeba/"
attributes_path = "/home/chrabasp/Download/list_attr_celeba.txt"


@click.group()
def cli():
    tf.logging.set_verbosity(tf.logging.INFO)


@cli.command()
@click.option('--dataset_path', required=True)
@click.option('--attributes_path', required=True)
@click.option('--batch_size', default=32)
@click.option('--img_size', default=32)
def train(dataset_path, attributes_path, batch_size, img_size):
    tf.logging.log(tf.logging.INFO, "Prepare dataset")
    data = Celeb(dataset_path, attributes_path, batch_size=batch_size, img_size=img_size)

    tf.logging.log(tf.logging.INFO, "Create model")
    model = ModelClassic(batch_size, img_size, steps=3, z_dim=128, k=1, y_dim=0)

    estimator = Estimator(model, data)

    tf.logging.log(tf.logging.INFO, "Start training")
    estimator.run_training(iterations=100000)


if __name__ == "__main__":
    cli()
