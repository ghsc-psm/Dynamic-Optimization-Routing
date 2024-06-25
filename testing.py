import os
from os.path import isfile, join

data_path = './data/COUNTRY/'
app_list = ['DRT', 'ALL', 'DRO']
# only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))
#              and (len(f.split(' ')) >= 2) 
#              and f.split(' ')[1].replace('.xlsx','') in app_list
#              and f[-4:] == 'xlsx']
# Print all files in the directory for debugging
all_files = os.listdir(data_path)
print("All files:", all_files)

# Filter the files step-by-step
filtered_files = []
for f in all_files:
    if isfile(join(data_path, f)):
        print(f"Checking file: {f}")  # Debug print
        parts = f.split(' ')
        if len(parts) >= 2:
            app_name = parts[1]
            if app_name in app_list and f.endswith('.xlsx'):
                filtered_files.append(f)
                print(f"File added: {f}")  # Debug print

only_files = filtered_files
print("Filtered files:", only_files)