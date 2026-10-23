"""Seven-dimensional IRIS using the shared standard-DH Drake backend."""
from scripts import spatial7R_model as model
from scripts import spatial4R_iris as shared


def build_plant(obstacles=None):
    return shared.build_plant(model.OBSTACLES if obstacles is None else obstacles, model.LINK_RADIUS, model)


def scene_metadata(obstacles):
    return shared.scene_metadata(obstacles, model)


def generate_regions(output, obstacles, seeds, coverage_radius=.16, max_regions=3):
    return shared.generate_regions(output, obstacles, max_regions=max_regions, seeds=seeds,
                                   coverage_radius=coverage_radius, model_module=model)
