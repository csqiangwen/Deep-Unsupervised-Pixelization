import os
import ntpath
from PIL import Image
from options.test_options import TestOptions
from data.custom_dataset_dataloader import CreateDataLoader
from model.pixelization_model import PixelizationModel


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

opt = TestOptions().parse()
opt.batchSize = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = PixelizationModel()
model.initialize(opt)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    img_path = model.get_image_paths()
    img_dir = os.path.join(opt.results_dir, '%s_%s' % (opt.phase, opt.which_epoch))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    print('%04d: process image... %s' % (i, img_path))
    for label, image_numpy in model.get_current_visuals_test().items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(img_dir, image_name)
        save_image(image_numpy, save_path)
