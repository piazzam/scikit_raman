import pandas as pd
from tqdm import tqdm
import os
import re

def importer(folder_path):
    """
    Import all the patient spectras contained in files in a folder. 

    Parameters
    ----------
    folder_path : string
        folder path to access to the spectral data. 

    Returns
    -------
    imported_data : pd.DataFrame
        dataframe with all spectra data contained in the folder. It is 
        formatted by the rules defined by our policy. See the documentation 
        for further information.

    """
    imported_data = pd.DataFrame()
    for filename in tqdm(os.listdir(folder_path)):
        patient_spectras = load_file(folder_path, filename)
        imported_data = imported_data.append(patient_spectras)
    return imported_data

def load_file(folder_path, filename,):
    """load the data from a single file. The file has to been formatted 
       followed the type of Raman BWS465-785. Further information in docs.

      Parameters:
      folder_path : string 
          the folder path to access the file
      filename : string 
          the filename to load
          
      Returns:
      patient_spectras (pd.DataFrame): 
          the spectras of the patient in the file. It follow the rules defined 
          by our policy. See the docs for further information.

     """
    imported_file = pd.read_csv(folder_path+ '/' + filename, sep=";")
    category, user = get_label_user(filename)
    user = user.replace('S', category)
    patient_spectras = pd.DataFrame(columns = ['user', 'spectra', 'x-axis', 'category', 'name', 'raw'])
    
    #raman_shift = imported_file['Raman Shift']
    raman_shift = list(imported_file['Raman Shift'].replace(",",".", regex=True).astype(float))
    imported_file.drop('Raman Shift', inplace = True, axis = 1)
    imported_file.drop('Pixel', inplace = True, axis = 1, errors='ignore')
    raw = False
    for column in imported_file.columns:
        if imported_file[column][0] == '\t':
            continue
        if 'Raw' in imported_file[column].name:
            raw = True
        else:
            raw = False
        patient_spectras = patient_spectras.append({'user': user, 'name':imported_file[column].name, 'raw':raw, 'spectra': list(imported_file[column].replace(",",".", regex=True).astype(float)), 'category': category, 'x-axis':raman_shift}, ignore_index = True)
    return patient_spectras
    
def get_label_user(filename):
    """
    Return the label of the patient and the name of patient. It extract the 
    information from the filename. The filename must follow our filename 
    policy.

    Parameters
    ----------
    filename : string
        a filename with spectral data. It must follow the policy.

    Returns
    -------
    String
        label of the patient
    String
        complete name of the patient

    """
    if '_20' in filename:
        filename = filename.replace('_20', '')
    f = filename.split('_10000')
    f = re.split('(\d+)', f[0])
    #return f[2], f[0]+f[1]
    return f[2], f[2]+'_'+f[1]


def save_as_pickle(df, filename):
    """
    Convert a dataframe in a pickle file. 

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame
    filename : string
        Path in which save the file

    """
    df.to_pickle(filename)
    
def get_raw_data(df):
    """
    Return a new DataFrame with only Raw Data. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted by our policies.

    Returns
    -------
    df_return : pd.DataFrame
        A DataFrame formatted by our policies with only raw data.

    """
    df_return = df[df['raw'] == True]
    return df_return

def get_dark_data(df):
    """
    Return a new DataFrame with only Dark Data. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted by our policies.

    Returns
    -------
    df_return : pd.DataFrame
        A DataFrame formatted by our policies with only dark data.

    """
    df_return = df[df['raw'] == False]
    return df_return

def category_to_label(df, conversion_dictionary):
    """
    Return a new dataframe with one more column named 'label'. This column 
    corresponds to a conversion from category to numerical label. The 
    conversion is made using the dictionary in input.
    
    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe that contained dataset defined as our policies. There is 
        category field. 
    conversion_dictionary : dict
        A dictionary that contains the rules for the conversion category to 
        label.

    Returns
    -------
    df : pd.DataFrame
        A complete DataFrame with the column label. 

    """
    df['label'] = [conversion_dictionary[el] for el in df['category']]
    return df    

def change_category_name(df, change_dictionary):
    """
    Change the name of one or more categories to another one choose by the 
    user. The conversion is made by the dictionary passed as input.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe formatted by our policies.
    change_dictionary : dict
        A dictionary that mapped old category names to new category names.

    Returns
    -------
    df : pd.DataFrame
        A dataframe formatted by our policies with new category names.

    """
    for el in change_dictionary:
        df.loc[df.category == el, 'category'] = change_dictionary[el]
    return df

def change_user_name_string(df, change_dictionary):
    """
    Change the string name of patients. The conversion is made by the passed 
    as input.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe formatted by our policy.
    change_dictionary : dict
        A dictionary that mapped old user names to new user names.

    Returns
    -------
    df : pd.DataFrame
        A dataframe formatted by our policies with new user names.

    """
    for el in df['user']:
        l = el.split('_')
        for d in change_dictionary:
            if d == l[0]:
                l[0] = change_dictionary[d]
                break
        string_name = l[0] + '_' + l[1]
        df.loc[df.user == el, 'user'] = string_name
    return df

def change_user_and_category(df, change_dictionary):
    """
    Change both category names and string of user names. It has a constraints 
    the two string must be the same.

    arameters
    ----------
    df : pd.DataFrame
        A dataframe formatted by our policy.
    change_dictionary : dict
        A dictionary that mapped old user names and categories 
        to new ones.

    Returns
    -------
    df : pd.DataFrame
        A dataframe formatted by our policies with new user names and 
        categories.

    """
    df = change_category_name(df, change_dictionary)
    df = change_user_name_string(df, change_dictionary)
    return df