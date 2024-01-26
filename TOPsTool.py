#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mechanize
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import numpy as np


# In[ ]:

TOPS_URL = "https://aphysics2.lanl.gov/apps/"
rlow="1.77827941e-17"
rmax="1.0e-3"
nr = '100'
ngroup = '50'
# lower energy in keV
egplow = '0.001'
# upper energy in keV
egphigh = '300'

TOPS_ELEM_ACCEPT = (
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
        "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
        "Fe", "Co", "Ni", "Cu", "Zn"
        )


def call_opdata(aMap: str,
                aTable: str,
                outputDir: str,
                jobs: int,
                opacityType: str = 'plank',
                emin: str = '0.001',
                emax: str = '300',
                ng: str = '50',
                SinglePoint: bool=False,
                outfilename: str = None,
                input_mass_frac: bool = False,
                den: str = '1',
                tem: str = '1'):
    """
    Main TOPS psuedo API call function. Will save results to outputDir with
    file format OP:IDX_X_Y_Z.dat where IDX is the ID of the composition (parallel
    to DSEP composition ID), X is the classical Hydrogen mass fraction, Y  is
    the classical Helium mass fraction, and Z is the classical metal mass
    fraction.

    Parameters
    ----------
        aMap : str
            Path to the list of classical compositions to be used. List should
            be given as an ascii file where ecach row is X,Y,Z
        aTable : str
            Path to chemical abundance table to be used as base composition.
        outputDir : str
            Path to directory save TOPS query results into
        jobs : int
            Number of threads to query TOPS webform on
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'

    Examples
    --------
    If you have some map of rescalings you would like to used at
    "./rescalings.dat" and you have a base composition in the correct form at
    "./comp.dat" then you can generate and cache the raw output for those rescalings of that composition using

    >>> call("./rescalings.dat", "./comp.dat", "./cache", 5)

    This will save the cache results to the folder ./cache (note that this
    folder must exist before calling call. Moreover, this will query using 5
    workers. You may increase this number to make call run faster; however,
    this will only work to a point. I find that around 20 workes is about the
    most that gives me any speed increase. This will somewhat depend on your
    computer though.

    By default this will query the plank /  mean grey opacities. However,
    if you want to query the multi-group opacities you can do so by passing
    opacityType='multi' to call. For example

    >>> call("./rescalings.dat", "./comp.dat", "./cache", 5, opacityType='multi')

    Note that pyTOPSScrape does not have any further functionality to parse
    the output of call for multi-group opacities. This is because the output
    format is different from the output format of the mean grey / plank
    opacities. If you would like to parse the output of the multi-group
    opacities you will need to do so yourself.
    """
    parsed = open_and_parse(aTable,input_mass_frac)
    pContents = parse_abundance_map(aMap)
    compList = list()
    for comp in pContents:
        zScale = comp[2]/parsed['AbundanceRatio']['Z']
        subComp = [
                ('H', comp[0]),
                ('He', comp[1])
                  ]
        for sym, data in parsed['RelativeAbundance'].items():
            if sym != 'H' and sym != 'He':
                subComp.append((sym, zScale * data['m_f']))

        compList.append(subComp)

    global egplow, egphigh, ngroup
    egplow = emin
    egphigh = emax
    ngroup = ng


    TOPS_query_async_distributor(compList, outputDir, njobs=jobs, opacityType=opacityType, outfilename=outfilename, SinglePoint=SinglePoint, den=den,tem=tem)


# In[ ]:


def TOPS_query_async_distributor(compList : list,
                                 outputDirectory : str,
                                 njobs : int = 10,
                                 nAttempts : int = 10,
                                 opacityType : str = 'plank',
                                 outfilename: str = None,
                                 SinglePoint: bool = False,
                                 den: str = '1',
                                 tem: str = '1'):
    """
    Distributes TOPS query jobs to different threads and gathers the results
    together. Writes out output.

    Parameters
    ----------
        comList : list
            3D list containing the mass frac for each element for each rescaled
            composition requested. For n compositions this should be of the shape
            (n, 30, 2). n compositions, 30 elements, then the first element of 
            the last axis is the elemental symbol (i.e. H, He, Li, Be, etc...)
            and the second element is the mass fraction.
        outputDirectory : str
            Path to directory to save TOPS query results to.
        njobs : int, default=10
            Number of concurrent jobs to allow at a time.
        nAttempts : int, default=10
            Number of time to retry TOPS query before failing out
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'
    """
    with tqdm(total=len(compList), desc=f"Querying on {njobs} threads") as pbar:
        with ThreadPoolExecutor(njobs) as executor:
            jobs = list()
            results = list()
            for i, subComp in enumerate(compList):
                jobs.append(executor.submit(query_and_parse, subComp, outputDirectory, i, opacityType=opacityType, outfilename=outfilename, SinglePoint=SinglePoint, den=den, tem=tem))
            for job in futures.as_completed(jobs):
                results.append(job.result())
                pbar.update(1)

    for table, filePath in tqdm(results, desc="Writing Query Results to Disk"):
        with open(filePath, 'w') as f:
            f.write(table)
            f.flush()


# In[ ]:

def query_and_parse(compList : list,
                    outputDirectory: int,
                    i: int,
                    nAttempts: int=10,
                    opacityType : str = 'plank',
                    outfilename: str = None,
                    SinglePoint: bool = False,
                    den: str = '1',
                    tem: str = '1'):
    """
    Async coroutine to query TOPS webform, parse the output, and write that
    to disk.

    Parameters
    ----------
        comList : list
            3D list containing the mass frac for each element for each rescaled
            composition requested. For n compositions this should be of the shape
            (n, 30, 2). n compositions, 30 elements, then the first element of
            the last axis is the elemental symbol (i.e. H, He, Li, Be, etc...)
            and the second element is the mass fraction.
        outputDirectory : str
            Path to write out results of TOPS webquery
        i : int
            Index of composition so file name can properly keep track of where
            it is, even in parallel processing.
        nAttempts : int, default=10
            Number of time to retry TOPS query before failing out
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'
    Raises
    ------
        ValueError : if opacityType is not 'plank' or 'multi'
    """
    if opacityType not in ['plank', 'multi']:
        raise ValueError(f"opacityType must be 'plank' or 'multi' not {opacityType}")

    mixString = format_TOPS_string(compList)
    X = compList[0][1]
    Y = compList[1][1]
    Z = 1 - X - Y

    assert Z >= 0.0

    Zsum = sum([x[1] for x in compList[2:]])

    if Zsum > 0.0:
        norm = Z/Zsum


        item = [(compList[2][0],compList[2][1]*norm)]
        if len(compList) > 3:
            for ele in compList[3:]:
                item = item + [(ele[0],ele[1]*norm)]


        mixString = format_TOPS_string(item)



    Xfmt = (int(X*1000) if int(X*1000) > 0 else 0)
    Yfmt = (int(Y*1000) if int(Y*1000) > 0 else 0)
    Zfmt = (int(Z*1000) if int(Z*1000) > 0 else 0)
    mixName = f"X{Xfmt} Y{Yfmt} Z{Zfmt}"


    tableHTML = TOPS_query(mixString, mixName, nAttempts, opacityType=opacityType, SinglePoint=SinglePoint, den=den, tem=tem)

    table = parse_table(tableHTML)


    if outfilename is None:
        filePath = f"{outputDirectory}/OP:{i}_{X}_{Y}_{Z}.dat"
        if opacityType == 'multi':
            filePath = f"{outputDirectory}/multi:{i}_{X}_{Y}_{Z}.dat"
    else:
        filePath = f"{outputDirectory}/"
        filePath = filePath+outfilename
        if opacityType == 'multi':
            filePath = f"{outputDirectory}/"
            filePath = filePath+outfilename


    return (table, filePath)


def TOPS_query(mixString: str, mixName: str, nAttempts: int, opacityType : str = 'plank', SinglePoint: bool=False, den: str = '1', tem: str = '1') -> bytes:
    """
    Query TOPS form and retry n times

    Parameters
    ----------
        mixString : string
            string in the form of: "massFrac0 Element0 massFrac1 Element1 ..."
            which will be submitted in the webform for mixture
        mixName : string
            name to be used in the webform
        nAttemptes : int
            How many times to reattempt after a failure.
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'

    Returns
    -------
        tableHTML : bytes
            Table queried from TOPS cite.

    """
    attempts = 0
    while attempts < nAttempts:
        try:
            if SinglePoint:
                tableHTML = submit_TOPS_form_single_T_rho(mixString, mixName, den, tem, opacityType=opacityType)
            else:
                tableHTML = submit_TOPS_form(mixString, mixName, opacityType=opacityType)
            break
        except mechanize.HTTPError as e:
            attempts += 1
            print(f"Unable to Query TOPS form for {mixName} "
                  f". Attempt number {attempts} of "
                  f"{nAttempts}")
            print(e)
    else:
        print(f"Unable to query TOPS form for {mixName}. Skipping!")
    return tableHTML


# In[ ]:



def submit_TOPS_form(
        mixString: str,
        mixName:str,
        massFrac: bool=True,
        opacityType='plank'
        ) -> bytes:
    """
    Open the Los Alamos opacity website, submit a given composition and then
    return the resultant table.

    Parameters
    ----------
        mixString : string
            string in the form of: "massFrac0 Element0 massFrac1 Element1 ..."
            which will be submitted in the webform for mixture
        mixName : string
            name to be used in the webform
        massFrac : bool, default=True
            Submit as massFrac instead of numberFrac
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'

    Returns
    -------
        tableHTML : bytes
            Table quired from TOPS cite.

    """
    br = mechanize.Browser()
    br.open(TOPS_URL)
    br.select_form(nr=0)
    if massFrac:
        br.form.find_control(name="fractype").value = ['mass']
    br.form['mixture'] = mixString
    br.form['mixname'] = mixName

    tlow = br.find_control(name="tlow", type="select").get("0.0005")
    tlow.selected = True
    tup = br.find_control(name="tup", type="select").get("100")
    tup.selected = True


    # These are the lowest and highest densities DSEP should need based
    #  on R = rho/T_6^3
    br.form['rlow'] = rlow
    br.form['rup'] =  rmax
    br.form['nr'] = nr

    if opacityType == 'multi':
        br.form.find_control(name="datype").value = ['groups']
        br.form['ngpengs'] = ngroup
        br.form['egplow'] = egplow
        br.form['egphigh'] = egphigh
    if opacityType == 'plank':
        br.form.find_control(name="datype").value = ['gray']


    # get to the first submission page
    r1 = br.submit()

    # get to the second submission page
    br.select_form(nr=0)
    r2 = br.submit()

    tableHTML = r2.read()
    br.close()
    return tableHTML


def submit_TOPS_form_single_T_rho(
        mixString: str,
        mixName:str,
        density: str,
        temperature: str,
        massFrac: bool=True,
        opacityType='multi'
        ) -> bytes:
    """
    Open the Los Alamos opacity website, submit a given composition and then
    return the resultant table.

    Parameters
    ----------
        mixString : string
            string in the form of: "massFrac0 Element0 massFrac1 Element1 ..."
            which will be submitted in the webform for mixture
        mixName : string
            name to be used in the webform
        massFrac : bool, default=True
            Submit as massFrac instead of numberFrac
        opacityType : str, default='plank'
            Type of opacity to query. Options are 'plank' and 'multi'
        density: string
            density used to get full frequency range opacity
        temperature: string
            temperature used to get full frequency range opacity in unit of keV

    Returns
    -------
        tableHTML : bytes
            Table quired from TOPS cite.

    """
    br = mechanize.Browser()
    br.open(TOPS_URL)
    br.select_form(nr=0)
    if massFrac:
        br.form.find_control(name="fractype").value = ['mass']
    br.form['mixture'] = mixString
    br.form['mixname'] = mixName

    br.form.find_control(name="tgrid").value = ['specific']

    br.form['temps'] = temperature


    br.form.find_control(name="rgrid").value = ['specific']

    br.form['dens'] = density


    if opacityType == 'multi':
        br.form.find_control(name="datype").value = ['groups']
        br.form['ngpengs'] = ngroup
        br.form['egplow'] = egplow
        br.form['egphigh'] = egphigh
    if opacityType == 'plank':
        br.form.find_control(name="datype").value = ['gray']


    # get to the first submission page
    r1 = br.submit()

    # get to the second submission page
    br.select_form(nr=0)
    r2 = br.submit()

    tableHTML = r2.read()
    br.close()
    return tableHTML



# In[ ]:



def open_and_parse(path,input_mass_frac):
    """
    Open and parse the contents of a chemical composition file

    Parameters
    ----------
        path : str
            Path to open file

    Returns
    -------
        parsed : dict
            Dictionary with two indexes.

                * Abundance Ratio
                    Includes the indexes:

                        - STD (*str*)
                        - [Fe/H] (*float*)
                        - [alpha/Fe] (*float*)
                        - [C/Fe] (*float*)
                        - [N/Fe] (*float*)
                        - [O/Fe] (*float*)
                        - [r/Fe] (*float*)
                        - [s/Fe] (*float*)
                        - C/O (*float*)
                        - X (*float*)
                        - Y (*float*)
                        - Z (*float*)

                * RelativeAbundance
                    Includes an index for each chemical symbol given in the
                    file format definition provided in the module
                    documentation. These are all floats.
    """

    contents = open_chm_file(path)
    parsed = parse(contents,input_mass_frac)

    return parsed


# In[ ]:


def parse_abundance_map(path : str) -> np.ndarray:
    """
    Parse Hydrogen, Helium, and metal mass fraction out of a csv where each row
    is one composition, the first column is X, second is Y, and the third is Z.
    Comments may be included in the file if the first non white space character
    on the line is a hash.

    Parameters
    ----------
        path : str
            Path to the abundance map. This should be an ascii file where each
            row contains X, Y, and Z (comma delimited with no white space).
            Each row will define one set of tables to be queried with the idea
            being that the entire file describes the entire set of tables to
            be queried.

    Returns
    -------
        pContents : np.ndarray(shape=(n,3))
            numpy array of all the compositions of length n where n is the
            number of rows whos first non white space character was not a hash.
            For a DSEP n=126. Along the second axis the first column is X, the
            second is Y, and the third is Z.

    """
    with open(path, 'r') as f:
        contents = f.read().split('\n')
        pContents = np.array([[float(y) for y in x.split(',')]
                               for x in contents
                              if x != '' and x.lstrip()[0] != '#'])
    return pContents


# In[ ]:


def parse(contents : list, input_mass_frac: bool) -> dict:
    """
    Parse chem file in the format described in the module documentation.

    The abuundance ratios and abundances on the first row are added to a dict
    under the key ['AbundanceRatio'] and sub indexed by the comments above each
    entry (Note that these are not read; rather, they are assumed to be the
    same in every file). The subsequent values (on all other rows) are added to
    the same dict under the key ['RelativeAbundance'] and sub indexed by their
    chemical symbols.

    Parameters
    ----------
        contents : list
            List of list of strings. The outter index selects the row, the
            inner index selected the column in the row and at each coordinate
            is a string which can be cast as a float. The one exception is that
            string at 0,0 is a charectar.

    Returns
    -------
        extracted : dict
            Dictionary with two indexes.

                * Abundance Ratio
                    Includes the indexes:

                        - STD (*str*)
                        - [Fe/H] (*float*)
                        - [alpha/Fe] (*float*)
                        - [C/Fe] (*float*)
                        - [N/Fe] (*float*)
                        - [O/Fe] (*float*)
                        - [r/Fe] (*float*)
                        - [s/Fe] (*float*)
                        - C/O (*float*)
                        - X (*float*)
                        - Y (*float*)
                        - Z (*float*)

                * RelativeAbundance
                    Includes an index for each chemical symbol given in the
                    file format from the module documentation. These are all
                    floats.

    """
    contentMap = [
            [
                'STD',
                '[Fe/H]',
                '[alpha/Fe]',
                '[C/Fe]',
                '[N/Fe]',
                '[O/Fe]',
                '[r/Fe]',
                '[s/Fe]',
                'C/O',
                'X',
                'Y',
                'Z'
            ],
            [
                ('H', 1.008),
                ('He', 4.003),
                ('Li', 6.941),
                ('Be', 9.012),
                ('B', 10.81),
                ('C', 12.01),
                ('N', 14.01),
                ('O', 16.00),
                ('F', 19.00),
                ('Ne', 20.18)
            ],
            [
                ('Na', 22.99),
                ('Mg', 24.31),
                ('Al', 26.98),
                ('Si', 28.09),
                ('P', 30.97),
                ('S', 32.07),
                ('Cl', 35.45),
                ('Ar', 39.95),
                ('K', 39.10),
                ('Ca', 40.08)
            ],
            [
                ('Sc', 44.96),
                ('Ti', 47.87),
                ('V', 50.94),
                ('Cr', 52.00),
                ('Mn', 54.94),
                ('Fe', 55.85),
                ('Co', 58.93),
                ('Ni', 58.69),
                ('Cu', 63.55),
                ('Zn', 65.38)
            ],
            [
                ('Ga', 69.72),
                ('Ge', 72.63),
                ('As', 74.92),
                ('Se', 78.97),
                ('Br', 79.90),
                ('Kr', 83.80),
                ('Rb', 85.47),
                ('Sr', 87.62),
                ('Y', 88.91),
                ('Zr', 91.22)
            ],
            [
                ('Nb', 92.91),
                ('Mo', 95.95),
                ('Tc', 98.00),
                ('Ru', 101.1),
                ('Rh', 102.9),
                ('Pd', 106.4),
                ('Ag', 107.9),
                ('Cd', 112.4),
                ('In', 1148),
                ('Sn', 118.7)
            ],
            [
                ('Sb', 121.8),
                ('Te', 127.6),
                ('I', 126.9),
                ('Xe', 131.3),
                ('Cs', 132.9),
                ('Ba', 137.3),
                ('La', 138.6),
                ('Ce', 140.1),
                ('Pr', 149.9),
                ('Nd', 144.2)
            ],
            [
                ('Pm', 145.0),
                ('Sm', 150.4),
                ('Eu', 152.0),
                ('Gd', 157.3),
                ('Tb', 158.9),
                ('Dy', 162.5),
                ('Ho', 164.9),
                ('Er', 167.3),
                ('Tm', 168.9),
                ('Yb', 173.04)
            ],
            [
                ('Lu', 175.0),
                ('Hf', 178.5),
                ('Ta', 180.9),
                ('W', 183.8),
                ('Re', 186.2),
                ('Os', 190.2),
                ('Ir', 192.2),
                ('Pt', 195.1),
                ('Au', 197.0),
                ('Hg', 200.6)
            ],
            [
                ('Tl', 204.4),
                ('Pb', 207.2),
                ('Bi', 209.0),
                ('Po', 209),
                ('At', 210),
                ('Rn', 222),
                ('Fr', 223),
                ('Ra', 226),
                ('Ac', 227),
                ('Th', 232)
            ],
            [
                ('Pa', 231),
                ('U', 238)
            ]
        ]
    extracted = {'AbundanceRatio': dict(), 'RelativeAbundance': dict()}
    for rowID, (row,target) in enumerate(zip(contents, contentMap)):
        for colID, (element, targetElement) in enumerate(zip(row, target)):
            if rowID == 0:
                if colID != 0:
                    element = float(element)
                extracted['AbundanceRatio'][targetElement] = element
            else:
                element = float(element)
                if input_mass_frac is False:
                    extracted['RelativeAbundance'][targetElement[0]] = {"a": element,
                                               "m_f": a_to_mfrac(element,
                                                                 targetElement[1],
                                                                 extracted['AbundanceRatio']['X']
                                                                )
                                              }
                else:
                    extracted['RelativeAbundance'][targetElement[0]] = {"a": element,
                                               "m_f": element}
    return extracted




# In[ ]:


def open_chm_file(path):
    """
    Open a chemical composition file (format defined in the module
    documentation). Split the contents by line then remove all lines which
    start with #. Finally split each line by both whitespace and commas.

    Parameters
    ----------
        path : str
            Path to file to open

    Returns
    -------
        contents : list
            List of list of strings. The outter index selects the row, the
            inner index selectes the column within the row.
    """
    with open(path, 'r') as f:
        contents = filter(lambda x: x != '', f.read().split('\n'))
        contents = filter(lambda x: x[0] != '#', contents)
    contents = [re.split(' |,', x) for x in contents]
    return contents


# In[ ]:


def a_to_mfrac(a,amass,X):
    """
    Convert :math:`a(i)` for the :math:`i^{th}` element to a mass fraction using the expression

    .. math::

        a(i) = \\log(1.008) + \\log(F_{i}) - \\left[\\log(X) + \\log(m_{i})\\right] + 12

    Or, equivilenetly, to go from :math:`a(i)` to mass fraction

    .. math::

        F_{i} = \\left[\\frac{X m_{i}}{1.008}\\right]\\times 10^{a(i)-12}

    Where :math:`F_{i}` is the mass fraction of the :math:`i^{th}` element,
    :math:`X` is the Hydrogen mass fraction, and :math:`m_{i}` is the ith
    element mass in hydrogen masses.

    Parameters
    ----------
        a : float
            :math:`a(i)` for the :math:`i^{th}` element. For example for He chem might
            be 10.93. For Hydrogen it would definititionally be 12.
        amass : float
            Mass of :math:`i^{th}` element given in atomic mass units.
        X : float
            Hydrogen mass fraction

    Returns
    -------
        mf : float
            Mass fraction of :math:`i^{th}` element.
    """
    mf = X*(amass/1.008)*10**(a-12)
    return mf


# In[ ]:


def parse_table(
        html: bytes
        ) -> str:
    """
    Parse the bytes table returned from mechanize into a string

    Parameters
    ----------
        html : bytes
            bytes table retuend from mechanize bowser at second TOPS submission
            form

    Returns
    -------
        table : string
            parsed html soruce in the form of a string
    """
    soup = BeautifulSoup(html, 'html.parser')

    # deal with line breaks
    table = soup.find('code').prettify().replace('<br/>', '')
    table = table.split('\n')

    # cut out the top and bottom lines of the table which dont matter
    table = [x for x in table[1:-2] if x.lstrip().rstrip() != '']

    # recombine the table into one string
    table = '\n'.join(table)
    return table


# In[ ]:



def format_TOPS_string(compList : list) -> str:
    """
    Format the composition list from pasrse_abundance_file into a string in the
    form that the TOPS web form expects for a mass fraction input.

    Parameters
    ----------
        compList : list
            composition list in the form of: [('Element', massFrac,
            numFrac),...]

    Returns
    -------
        TOPS_abundance_string : string
            string in the form of: "massFrac0 Element0 massFrac1 Element1 ..."

    """
    TOPS_abundance_string = ' '.join([
        f"{x[1]:0.10f} {x[0]}"
        for x in compList
        if x[0] in TOPS_ELEM_ACCEPT
                                    ])
    return TOPS_abundance_string

