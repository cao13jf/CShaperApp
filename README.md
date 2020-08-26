## CShaper

### Parameters
* All parameters are saved in `./ConfigMemb/test_edt_discrete.txt`, which is read by `./test_edt.py` through 
    ```python
    config = parse_config(config_file)  # inside the `./test_edt.py`
    ```
    For the GUI app, you may need to read all paras interactively instead of reading the file. It means that 
    you need to design the `*.ui` file by adding or deleting components. 
    
* Please note that the parameter `data_names` is a multi-choice which lists all 1th-level folder names under `data_folder`.
For example, in the following example data, `data_names` should be `181210plc1p1` / `181210plc1p2` / `181210plc1p3`, from which
the user can choose one or multiple embryos.
    
### Example data
* The example data can be downloaded with this [link](https://portland-my.sharepoint.com/:f:/g/personal/jfcao3-c_ad_cityu_edu_hk/Ej_eY_L11SBMiUJk_S4_UMYBaamvJMpQAaAECcyU6bmWxQ?e=1zpsVC), 
which includes three embryos. Please put `./ExampleData` to the same root folder as this code repository. 
The template parameter settings for this data should be
    ```text
    [para]
    
    # segmentation
    data_folder         = Data/MembTest
    embryo_names        = [181210plc1p1, 181210plc1p2, 181210plc1p3]  # should list all embryos in the folder
    max_time            = 150
    save_folder         = ResultCell
    batch_size          = 2
    nucleus_as_seeds    = False
    nucleus_filter      = True
    
    # shape analysis
    image_size          = [68, 712, 512]
    first_run           = False
    analysis_embryo_names        = [181210plc1p3]
    ```

### How to test the functionality
1. Download 1). `Example data` to `./`; 2). these [files](https://portland-my.sharepoint.com/:f:/g/personal/jfcao3-c_ad_cityu_edu_hk/Er7DRPjxdBRGrwB7X3L7yS0B9j5FPlmUCW9ahDLOa1qW9w?e=Ig6g9b) to `./ShapeUtil/`; 
3). trained [model](https://portland-my.sharepoint.com/:f:/g/personal/jfcao3-c_ad_cityu_edu_hk/EvuLuIAH7u5FteDDcitG7eYB0QIlZoXD_keas05ENqMYvw?e=l8O6na) to `./ModelCell/` So the final folder structure should be 
    ```text
    CShaperApp/: code root folder
       |--ExampleData/: example data downloaded from the link
           |--181210plc1p1/: 1st embryo folder
               |******
           |--181210plc1p2/: 2nd embryo folder
               |******
           |--*****
       |--ModelCell/: trained model
           |--*.cpkt
           |--*****
       |--ShapeUtil/:
           |--number_dictionary.csv
           |--number_dictionary.txt
           |--******
       |--*** : (some folders that can be cloned from this repository are not listed here)
    ```
2. Inside the GUI app, `test_edt.py` and `shape_analysis.py` should be able to run consecutively after receiving the 
parameters. s