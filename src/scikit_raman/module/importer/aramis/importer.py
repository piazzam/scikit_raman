import os
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

def importer_new(folder, excel_file, base_code, label_parsing_function='get_label_user'):
    df = importer(folder, label_parsing_function=label_parsing_function)

    df_phen = pd.read_excel(excel_file, sheet_name='Foglio1')

    user = []
    category = []
    for i, row in df.iterrows():
        if label_parsing_function == get_label_user:
            code = str(base_code)+'-'+row['user'].split('_')[1]
        elif label_parsing_function == get_label_user_v2:
            code = row['user']
        else:
            raise Exception("Parsing function file not defined")

        row_phen = df_phen[df_phen['ID REDCAP'] == code]
        if row_phen['Place_code'].iloc[0] != base_code:
            raise Exception("Error code")
        is_acquired = row_phen['Acquisition'].iloc[0]
        diagnosis = row_phen['Diagnosis'].iloc[0]
        if diagnosis == "":
            continue
        if is_acquired == 'T':
            if diagnosis == 'Asthma (control group)':
                new_user_string = 'ASMA_' + code
                new_category = 'ASMA'
            elif diagnosis == 'Healthy subjects (control group)':
                new_user_string = 'CTRL_' + code
                new_category = 'CTRL'
            else:
                new_user_string = 'BPCO_' + code
                new_category = 'BPCO'
        user.append(new_user_string)
        category.append(new_category)

    df['user'] = user
    df['category'] = category

    return df
    #df.to_pickle(output_file)

def importer_farmaci(folder_path, cat_type = True):
    imported_data = pd.DataFrame()
    for filename in tqdm(os.listdir(folder_path)):
        user = filename.split('.')[0]
        if cat_type:
            cat = filename.split('.')[0].split('_')[0]
        else:
            cat = filename.split('.')[0].split('_')[1].replace(' ', '_')
        patient_spectras = pd.DataFrame(columns=['user', 'name', 'raw', 'spectra', 'category', 'x-axis'])
        data = open(folder_path + '/' + filename)
        rows = [line.split('\t') for line in data]
        raw = True
        for j in range(len(rows) - 1):
            if (rows[0][1] == ''):
                raman_shift = [float(el) for el in rows[0][2:]]
                line = [float(el) for el in rows[j + 1][2:]]
            else:
                raman_shift = [float(el) for el in rows[0][1:]]
                line = [float(el) for el in rows[j + 1][1:]]
            patient_spectras = patient_spectras.append(
                {'user': user, 'name': 'Raw', 'raw': raw, 'spectra': line, 'category': cat, 'x-axis': raman_shift},
                ignore_index=True)
        imported_data = imported_data.append(patient_spectras)
    imported_data = imported_data.reset_index(drop=True)
    return imported_data

def importer(folder_path, label_parsing_function='get_label_user'):
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
        cat = os.path.basename(os.path.normpath(folder_path))
        cat = cat.split('_')[0]
        f = filename.split('_Allu')
        f = re.split('(\d+)',f[0])
        patient_spectras = load_single_file(folder_path, filename, cat, label_parsing_function=label_parsing_function)
        imported_data = pd.concat([imported_data, patient_spectras], ignore_index=True)
        #imported_data = imported_data.append(patient_spectras)
    imported_data = imported_data.reset_index(drop=True)
    return imported_data



# def load_single_file(folder_path, filename, cat):
#     """load the data from a single file. The file has to been formatted
#        followed the type of Raman Aramis. Further information in docs.
#
#       Parameters:
#       folder_path : string
#           the folder path to access the file
#       filename : string
#           the filename to load
#       cat: string
#           the category extracted from the folder path
#
#       Returns:
#       patient_spectras (pd.DataFrame):
#           the spectras of the patient in the file. It follow the rules defined
#           by our policy. See the docs for further information.
#
#      """
#     print("load_single_file")
#     l,u = get_label_user(filename)
#     if "FARM" in u:
#       u = u.replace("FARM", cat)
#       u = u + 'F'
#       l = cat
#     elif ('20' in u.split('_')[1]) and u.split('_')[1] != '20':
#         u = u.replace('20', 'H')
#     if len(u.split('_')) > 2:
#       return None
#     if l != cat:
#       u = u.replace(l, cat)
#       l = cat
#     patient_spectras = pd.DataFrame(columns=['user', 'name', 'raw', 'spectra', 'category', 'x-axis'])
#     data = open(folder_path+'/'+filename)
#     rows = [line.split('\t') for line in data]
#     raw = True
#     len(rows)
#     for j in range(len(rows)-1):
#         if(rows[0][1] == ''):
#             #raman_shift = rows[0][2:]
#             raman_shift = [float(el) for el in rows[0][2:]]
#             #raman_shift = np.array(raman_shift)
#             line = [float(el) for el in rows[j+1][2:]]
#             #line = np.array(line)
#         else:
#             raman_shift = [float(el) for el in rows[0][1:]]
#             #raman_shift = np.array(raman_shift)
#             #raman_shift = rows[0][1:]
#             line = [float(el) for el in rows[j+1][1:]]
#             #line = np.array(line)
#         patient_spectras = pd.concat([patient_spectras,
#                                       pd.DataFrame({'user': u, 'name':'Raw', 'raw':raw, 'spectra': line, 'category': l, 'x-axis':raman_shift})] ,
#                                      ignore_index = True)
#         #patient_spectras = patient_spectras.append({'user': u, 'name':'Raw', 'raw':raw, 'spectra': line, 'category': l, 'x-axis':raman_shift}, ignore_index = True)
#     return patient_spectras


def load_single_file(folder_path, filename, cat, label_parsing_function):
    """load the data from a single file. The file has to been formatted
       followed the type of Raman Aramis. Further information in docs.

      Parameters:
      folder_path : string
          the folder path to access the file
      filename : string
          the filename to load
      cat: string
          the category extracted from the folder path

      Returns:
      patient_spectras (pd.DataFrame):
          the spectras of the patient in the file. It follow the rules defined
          by our policy. See the docs for further information.

     """
    l,u = label_parsing_function(filename)
    if l == "":
        l = "X"
        u = "X"+u
    if "FARM" in u:
        u = u.replace("FARM", cat)
        u = u + 'F'
        l = cat
    if '_' in u:
        if ('20' in u.split('_')[1]) and u.split('_')[1] != '20':
            u = u.replace('20', 'H')
    if len(u.split('_')) > 2:
        return None
    if l != cat:
        u = u.replace(l, cat)
        l = cat
    #patient_spectras = pd.DataFrame(columns=['user', 'name', 'raw', 'spectra', 'category', 'x-axis'])
    data = open(folder_path + '/' + filename)
    rows = [line.split('\t') for line in data]
    raw = True
    users = []
    names = []
    spectras = []
    categories = []
    x_axis_mul = []
    raws = []
    for j in range(len(rows) - 1):
        if (rows[0][1] == ''):
            # raman_shift = rows[0][2:]
            raman_shift = [float(el) for el in rows[0][2:]]
            # raman_shift = np.array(raman_shift)
            line = [float(el) for el in rows[j + 1][2:]]
            # line = np.array(line)
        else:
            raman_shift = [float(el) for el in rows[0][1:]]
            # raman_shift = np.array(raman_shift)
            # raman_shift = rows[0][1:]
            line = [float(el) for el in rows[j + 1][1:]]
            # line = np.array(line)
        users.append(u)
        names.append('Raw')
        spectras.append(line)
        categories.append(l)
        x_axis_mul.append(raman_shift)
        raws.append(raw)
        #patient_spectras = pd.concat([patient_spectras,
        #                              pd.DataFrame(
        #                                  {'user': u, 'name': 'Raw', 'raw': raw, 'spectra': line, 'category': l,'x-axis': raman_shift})],
        #                             ignore_index=True)
        #patient_spectras = patient_spectras.append(
        #    {'user': u, 'name': 'Raw', 'raw': raw, 'spectra': line, 'category': l, 'x-axis': raman_shift},
        #    ignore_index=True)
    d = {'user':users, 'name':names, 'raw':raws, 'spectra':spectras, 'category':categories, 'x-axis':x_axis_mul}
    patient_spectras = pd.DataFrame(d)
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
    f = filename.split('_Allu')
    f = re.split('(\d+)',f[0])
    #return f[2], f[0]+f[1]
    return f[2], f[2]+'_'+f[1]

def get_label_user_v2(filename):
    f = filename.split('_Allu')
    return 'null', f[0]

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
        A DataFrame formatted according to our policy.

    Returns
    -------
    df_return : pd.DataFrame
        A DataFrame formatted according to our policy with only raw data.

    """
    df_return = df[df['raw'] == True]
    df_return = df_return.reset_index()
    return df_return

def get_dark_data(df):
    """
    Return a new DataFrame with only Dark Data. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy.

    Returns
    -------
    df_return : pd.DataFrame
        A DataFrame formatted according to our policy with only dark data.

    """
    df_return = df[df['raw'] == False]
    df_return = df_return.reset_index()
    return df_return

def category_to_label(df, conversion_dictionary):
    """
    Return a new dataframe with a more column named 'label'. This column 
    corresponds to a conversion from category to numerical label. The 
    conversion is made using the dictionary in input.
    
    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe that contained dataset formatted according to our policy. 
        The Dataframe must have category field. 
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
    user.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe formatted according to our policy.
    change_dictionary : dict
        A dictionary that mapped old category names to new category names.

    Returns
    -------
    df : pd.DataFrame
        A dataframe formatted according to our policy with new category names.

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
        A dataframe formatted according to our policy with new user names.

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
        A dataframe formatted according to our policy.
    change_dictionary : dict
        A dictionary that mapped old user names and categories 
        to new ones.

    Returns
    -------
    df : pd.DataFrame
        A dataframe formatted according to our policy with new user names and 
        categories.

    """
    df = change_category_name(df, change_dictionary)
    df = change_user_name_string(df, change_dictionary)
    return df