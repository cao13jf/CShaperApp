## CShaper

### Parameters
* All parameters are saved in `./ConfigMemb/test_edt_discrete.txt`, which is read by `./combine_slice.py`, `./test_edt.py`
or `./shape_analysis.py` with `configparser` library. 

    For the GUI app, you may need to read all paras interactively instead of reading the file. It means that 
    you need to design the `*.ui` file by adding or deleting components [Example GUI](https://pan.baidu.com/s/1fe27UmV16mvYk0qYDdv_tw) (PW：0000). 
    
* Please note that the parameter `data_names` is a multi-choice which lists all 1th-level folder names under `data_folder`.
For example, in the following example data, `data_names` should be `181210plc1p1` / `181210plc1p2` / `181210plc1p3`, from which
the user can choose one or multiple embryos.
    
### Example data
* The example data can be downloaded with this [link](https://pan.baidu.com/s/1zpcicT928--WV5N2qbY00A) (PW: 7omp)
which includes two embryos. Please put `./ExampleData` to the same root folder as this code repository. 
The template parameter settings for this data should be
    ```text
    # parameters for `combine_slice.py`
    embryo_names        = [181210plc1p1, 181210plc1p2]
    xy_resolution       = 0.09
    z_resolution        = 0.42
    max_time            = 50
    reduce_ratio        = 0.50
    stack_folder        = Data/MembTest
    num_slice           = 68
    raw_folder          = ./Data/MembRaw
    lineage_file        = ./Data/MembRaw/181210plc1p1/aceNuc/CD181210plc1p1.csv
    
    # parameters for `test_edt.py`
    # <-- stack_folder        = Data/MembTest
    # <-- embryo_names        = [181210plc1p1]
    # <-- max_time            = 150
    save_folder         = ResultCell
    batch_size          = 2
    # following two choices should be either True|False or False|True
    nucleus_as_seed     = False
    nucleus_filter      = True
    
    # parameters for `shape_analysis.py`
    # <-- num_slice           = 68
    # <-- xy_resolution       = 0.09
    # <-- raw_folder          = ./Data/MembRaw
    # <-- embryo_names        = [181210plc1p3]
    # <-- stack_folder        = Data/MembTest
    first_run                 = False
    ```

### How to test the functionality
1. Download 1). `Example data` to `./Data/MembRaw`; 2). these [files](https://pan.baidu.com/s/1PSRvj7n6s8rJnzzG43X16g) (PW: o2zy) to `./ShapeUtil/`; 
3). trained [model](https://pan.baidu.com/s/1OjX4E-z2ZecOsvGaoctnRw) (pw: 6t6v) to `./ModelCell/`. So the final folder structure should be 
    ```text
    CShaperApp/: code root folder
       |--Data/: example data downloaded from the link
           |--MembRaw/: raw slice data
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
2. Inside the GUI app, `combine_slice.py`, `test_edt.py` and `shape_analysis.py` should be able to run independently or 
consecutively after receiving the parameters.