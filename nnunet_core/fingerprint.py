import warnings
import multiprocessing
import os
from time import sleep
from typing import List, Type,Union
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from tqdm import tqdm
from imageio.base_reader_writer import BaseReaderWriter
from preprocessing.cropping.cropping import crop_to_nonzero
from imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from paths import nnUNet_raw, nnUNet_preprocessed
from utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from utilities.utils import get_filenames_of_train_images_and_targets

class DatasetFingerprintExtractor(object):
    def __init__(self,dataset_name_or_id:Union[str,int],num_processes: int=8,verbose :bool=False):
        dataset_name=maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw,dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder,'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder,self.dataset_json)
        self.num_foreground_voxels_for_intensitystats = 10e7
    
    @staticmethod
    def collect_foreground_intensities(segmentation:np.ndarray,images:np.ndarray,seed:int=1234,num_samples:int=10000):

        assert images.ndim == 4 and segmentation.ndim == 4
        assert not np.any(np.isnan(segmentation)),"Segmentation contains NaN values :-("
        assert not np.any(np.isnan(images)),"Images contains NaN values :-("

        rs = np.random.RandomState(seed)
        intensities_per_channel = []
        intensity_statistics_per_channel = []
        foreground_mask = segmentation[0]>0
        percentiles = np.array((0.5,50.0,99.5))
        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_mask)

            intensities_per_channel.append(rs.choice(foreground_pixels,num_samples,replace=True)if num_fg>0 else [])
            mean,median,mini,maxi,percentile_99_5,percentile_00_5 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            if num_fg >0 :
                percentile_00_5,median,percentile_99_5 =  np.percentile(foreground_pixels,percentiles)
                mean = np.mean(foreground_pixels)
                mini = np.min(foreground_pixels)
                maxi = np.max(foreground_pixels)
            intensity_statistics_per_channel.append({
                'mean':mean,
                'median':median,
                'min':mini,
                'max':maxi,
                'percentile_99_5':percentile_99_5,
                'percentile_00_5':percentile_00_5,
            })
        return intensities_per_channel,intensity_statistics_per_channel
    
    @staticmethod
    def analyse_case(image_files:List[str],segmentation_file:str,reader_writer_class:Type[BaseReaderWriter],num_samples:int=10000):
        rw= reader_writer_class()
        images,properties_images = rw.read_images(image_files)
        segmentation,properties_seg = rw.read_seg(segmentation_file)
        data_cropped,seg_cropped,bbox = crop_to_nonzero(images,segmentation)
        foreground_intensities_per_channel,foreground_intensity_stas_per_channel = DatasetFingerprintExtractor.collect_foreground_intensities(seg_cropped,data_cropped,num_samples=num_samples)
        spacing = properties_images['spacing']
        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop)/np.prod(shape_before_crop)
        return shape_after_crop,spacing,foreground_intensities_per_channel,foreground_intensity_stas_per_channel,relative_size_after_cropping
    
    def run(self,overwrite_existing:bool=False)->dict:
        preprocesses_output_folder = join (nnUNet_preprocessed,self.dataset_name)
        maybe_mkdir_p(preprocesses_output_folder)
        properties_file = join(preprocesses_output_folder,'dataset_fingerprint.json')
        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0])
            num__foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats//len(self.dataset))
            r=[]
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in self.dataset.keys():
                    r.append(p.starmap_async(DatasetFingerprintExtractor.analyse_case,((self.dataset[k]['images'],self.dataset[k]['label'],reader_writer_class,num__foreground_samples_per_case),)))
                remaining= list(range(len(self.dataset)))
                workers = [j for j in p._pool]
                with tqdm(desc=None,total = len(self.dataset),disable=self.verbose) as pbar:
                    while len(remaining)>0:
                        all_alive= all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError('Some bg worker is dead')
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)
            results = [i.get()[0] for i in r]
            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [np.concatenate([r[2][i] for r in results])for i in range(len(results[0][2]))]
            median_relative_size_after_cropping = np.median([r[4] for r in results],0)
            num_channels = len(self.dataset_json['channel_names'].keys() if 'channel_names' in self.dataset_json.keys() else self.dataset_json['modality'].keys())
            intensity_statistics_per_channel = {}
            percentiles = np.array((0.5,50.0,99.5))
            for i in range(num_channels):
                percentiles_00_5,median,percentile_99_5 = np.percentile(foreground_intensities_per_channel[i],percentiles)
                intensity_statistics_per_channel[i] = {'mean' :float(np.mean(foreground_intensities_per_channel[i])),'median':float(median),'std':float(np.std(foreground_intensities_per_channel[i])),'min':float(np.min(foreground_intensities_per_channel[i])),'max':float(np.max(foreground_intensities_per_channel[i])),'percentile_99_5':float(percentile_99_5),'percentile_00_5':float(percentiles_00_5),}
            fingerprint = {
                "spacings":spacings,
                "shapes_after_crop":shapes_after_crop,
                "foreground_intensities_per_channel":intensity_statistics_per_channel,
                "median_relative_size_after_cropping":median_relative_size_after_cropping
            }
            try:
                save_json(fingerprint,properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint
    

if __name__=='__main__':
    dfe = DatasetFingerprintExtractor(2,8)
    dfe.run(overwrite_existing=False)