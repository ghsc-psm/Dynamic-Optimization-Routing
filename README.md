# Distribution Route Optimizer (DRO) Solution

## Solutions contained in this repository

1. Standard Delivery Route Optimizer
2. Fleet Restriction Generation Template and Script
3. Batch Processing Script for seperated DRT and Order Evaluation Template files
4. Truck Assignment Optimization to ensure most optimal truck assignments (Posthoc Analysis; Ensures Optimal Assignment for Predefined_Routes)

## Setting Configurations for VRP Optimizer

Prior to running the streamlit application, configure `country_cfg.toml` with country specific information including:

1. Country Name (country_name)
2. Tool Display Name (tool_title)
3. Currency (currency)

For `drt_functionaility`, list either `True` or `False` next to each specific entry to enable or disable the particular functionality of the application when the app is run.

For `dro_specs` and `baseline`, list either `True` or `False` for the specific optimization functionalities to either include or not include toggles for an optimization restriction to be customizable (i.e. allow users to constrain or not constrain based on specifc parameters in the application itself).

## Important Checks Prior to Running any Solution

Prior to running, it is essential to verify the following:

1. There exists a folder in `./data` that matches the country name specified in `country_cfg.toml` with the first letter of each word captialized. This folder should be modeled after the sample `./data/COUNTRY/` folder
2. Ensure the following tables are filled out, Facility, Distance, Time, Fleet, and Fleet Exclusions.
3. All files in folder specified above are accurate; this includes the `warehouse_mapping.json` file.

## How to Run Standard Delivery Route Optimizer

To run locally:

1. Git clone the repository.

2. Create virtual environment and activate it
    - `python -m venv .venv`
    - `source .venv/Script/activate`

    Note: Once the virtual environment is created, the environment must be activated each time starting a new session.

3. Install all requirements: 
    - `pip install -r requirements.txt`
    - Go into the lib folder and install psm library: `pip install psm-X.X.X-py3-none-any.whl`

4. Add a folder called `filestore`.

5. Ensure `country_cfg.toml` is updated with information above:

6. Ensure that you filling the `COUNTRY_DRT_Files.xlsx` and have all the DRT files directly in the `data/COUNTRY` folder.

7. Run the app in terminal!
    - `streamlit run app.py`

## Replicating Predefined Routes

If historical route information is available, it is possible to generate a baseline cost associated with these historical routes using the Predefined Routes. A baseline scenario optimization can be conducted by setting the `predefined_routes` option to `True` in the `baseline` portion of the `country_config.toml` file. This will allow for the toggle to become available in the Order Evaluation portion of the application. With this toggle on, the application will find the optimal route for predefined routes listed in the Order Evaluation File that is uploaded. This in most cases will provide the optimal solution; however, there are few circumstances where this will not be the case:

1. A predefined route is separated into two or more route as this would be more optimal. If this occurs, a message will be displayed to the user noting that this has in fact occurred. Additional steps would be necessary to correct this such as manually changing the fix cost in the DRT Fleet Tab and re-running the application.  *Note: this is a rare occurence that is likely to not occur.*
2. If all vehicles of a specific type are used prior to completing the predefined_routes optimization, the `truck_optimize` optimization will ensure the most optimal assignment of trucks to specific routes in order to ensure that the most optimal route is presented.

## Generating the Fleet Restrictions:

To generate the fleet restrictions for a DRT file, use and fill out sample template: `Fleet_Restriction_Template.xlsx`. In this file, you will find instructions on how to complete this template. Once this is complete saved the updated file.

Open `fleet_exclusion.py` in VSCode and update `fname` (located at the bottom of the code) with the path to the completed Fleet Restriction Template. Then open terminal and run:

`python fleet_exclusion.py`

This will create a new xlsx file. The first sheet in the workbook can be copied into the DRT Fleet Restriction sheet. The second sheet contains quality checks to ensure that the volume at the facility is not greater than the volume capacity of the largest vehicle that can get to the facility. This check is to ensure that the DRO Optimizer does not encounter problems when working to produce optimization results.

## Running the Batch Optimize Process:

To run the batch optimization, run the following command:

`streamlit run batch_process.py`

This will launch a streamlit page that appears similar to the app.py page. The key difference stems from the fact that you are able to upload multiple DRT and Order Evaluation Template Files. In fact, for each DRT file added, you can add one or more Order Evaluation Template files to optimize. Once the DRT and Order Evaluation Template file(s) are uploaded, select `Confirm DRT and Order Evaluation Template File Selection`. Once this is done, the selected files will be listed below in the `Selected DRT and Order Evaluation Template Files for optimization` section. More DRT and Order Evaluation Template files can be added and commited in a similar manner. If there is a mistake in the uploaded files, select `Clear all Optimizations Previously Selected` and reupload files to process. Once ready to run the Optimizations, select the `Confirm files for optimization` button and the app will display the optimization parameters and have an option to run the optimization similar to the `app.py` application. Note: The Parameters specified in the DRT file will take priority over any parameters manually set in the application when the optimization is running.

## Running the Truck Optimization Process:

To run the Truck Optimization, first open `truck_optimize.py` in VSCode and update `scenario_file_path` (located at the bottom of the code) with the path to the scenario where there is interest in running the post-hoc truck optimization. Save these changes.

Finally, run the following command:

`streamlit run truck_optimize.py`

Once the optimization is complete, it will say so and state that the file has been saved.

### Key Files:

**Python Files**

- `app.py`
- `app_admin.py`
- `app_login.py`
- `app_drorders.py`
- `app_upload.py`
- `app_refine.py`
- `app_optimize.py`
- `app_review.py`
- `app_release_notes.py`
- `DRO.py`
- `SessionState.py`
- `Scenario.py`
- `map_utils.py`
- `matrix_utils.py`

**Other Files**
- `country_cfg.toml`
- `credentials.ini`
- `requirements.txt`
