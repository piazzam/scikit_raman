import yaml
def parse_yaml_file(file):
    """
    Parse a configuration file in .yaml format and return its content in a dictionary
    Parameters
    ----------
    file : string
        Path to configuration file.

    Returns
    -------
    dict
        Dictionary with .yaml configurations.
    """
    with open(file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def extract_drugs_list_polvere(ds_farmaco):
    """
    Load the spectra from powder drugs

    Parameters
    ----------
    ds_farmaco : scikit_raman.Dataset
        Drugs dataset

    Returns
    -------
    list
        List containing spectra for each drug
    """
    ds_farmaco = ds_farmaco.search_by_category_name('polvere')
    ds_farmaco = ds_farmaco.create_mean_spectra()
    ds_farmaco_uno = ds_farmaco.search_by_name('farmaco1_polvere')
    ds_farmaco_uno.spectra = ds_farmaco_uno.spectra[0][6:]
    ds_farmaco_due = ds_farmaco.search_by_name('farmaco2_polvere')
    ds_farmaco_due.spectra = ds_farmaco_due.spectra[0][6:]
    ds_farmaco_tre = ds_farmaco.search_by_name('farmaco3_polvere')
    ds_farmaco_tre.spectra = ds_farmaco_tre.spectra[0][6:]
    ds_farmaco_quattro = ds_farmaco.search_by_name('farmaco4_polvere')
    ds_farmaco_quattro.spectra = ds_farmaco_quattro.spectra[0][6:]
    ds_farmaco_cinque = ds_farmaco.search_by_name('farmaco5_polvere')
    ds_farmaco_cinque.spectra = ds_farmaco_cinque.spectra[0][6:]
    ds_farmaco_sei = ds_farmaco.search_by_name('farmaco6_polvere')
    ds_farmaco_sei.spectra = ds_farmaco_sei.spectra[0][6:]
    drugs = [ds_farmaco_uno.spectra, ds_farmaco_due.spectra, ds_farmaco_tre.spectra, ds_farmaco_quattro.spectra,
             ds_farmaco_cinque.spectra, ds_farmaco_sei.spectra]
    return drugs

def extract_drugs_list_fisio(ds_farmaco):
    """
    Load the spectra from water solution drugs

    Parameters
    ----------
    ds_farmaco : scikit_raman.Dataset
        Drugs dataset

    Returns
    -------
    list
        List containing spectra for each drug
    """
    ds_farmaco = ds_farmaco.search_by_category_name('in_fisio')
    ds_farmaco = ds_farmaco.create_mean_spectra()
    ds_farmaco_uno = ds_farmaco.search_by_name('farmaco1_in fisio')
    ds_farmaco_uno.spectra = ds_farmaco_uno.spectra[0][6:]
    ds_farmaco_due = ds_farmaco.search_by_name('farmaco2_in fisio')
    ds_farmaco_due.spectra = ds_farmaco_due.spectra[0][6:]
    ds_farmaco_tre = ds_farmaco.search_by_name('farmaco3_in fisio')
    ds_farmaco_tre.spectra = ds_farmaco_tre.spectra[0][6:]
    ds_farmaco_quattro = ds_farmaco.search_by_name('farmaco4_in fisio')
    ds_farmaco_quattro.spectra = ds_farmaco_quattro.spectra[0][6:]
    ds_farmaco_cinque = ds_farmaco.search_by_name('farmaco5_in fisio')
    ds_farmaco_cinque.spectra = ds_farmaco_cinque.spectra[0][6:]
    ds_farmaco_sei = ds_farmaco.search_by_name('farmaco6_in fisio')
    ds_farmaco_sei.spectra = ds_farmaco_sei.spectra[0][6:]
    drugs = [ds_farmaco_uno.spectra, ds_farmaco_due.spectra, ds_farmaco_tre.spectra, ds_farmaco_quattro.spectra,
             ds_farmaco_cinque.spectra, ds_farmaco_sei.spectra]
    return drugs