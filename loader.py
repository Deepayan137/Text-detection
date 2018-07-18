from lxml import etree
import pdb
from tqdm import *
import csv
import pandas as pd
import pdb
import cv2
import os
def line_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    olines = []
    for atom in atoms:
        i = int(atom["LineNo"])
        olines.append(lines[i-1])
    return (olines, atoms)

def word_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    words = list(map(lambda x: x.split(), lines))
    owords = []
    for atom in atoms:
        i, j = list(map(lambda x: int(atom[x]), ["LineNo", "SerialNo"]))
        owords.append(words[i-1][j-1])
    return (owords, atoms)

def parse_ocr_xml(xml_file):
    with open(xml_file, encoding='utf-8') as f:
        #print("Read", xml_file)
        root = etree.parse(f)
        rows = root.xpath("row")

        # Subfunction to extract from a row
        extract = lambda field: (field.get("name"), field.text)
        extract_field = lambda x: dict(map(extract, x.xpath("field")))
        export = list(map(extract_field, rows))
        return export


def group(text_data, units):
    udict = {}
    required_keys = ["ImageLoc", "Text"]
    udict["page"] = {}
    for entry in text_data:
        if not "BookCode" in udict:
            udict["BookCode"] = entry["BookCode"]
        pno = int(entry["PageNo"])
        if pno not in udict["page"]:
            udict["page"][pno] = dict((key, entry[key]) for key in required_keys)
            udict["page"][pno]["units"] = []

    for unit in units:
        pno = int(unit["PageNo"])
        udict["page"][pno]["units"].append(unit)

    # Return ultimate dict
    return udict



def images_and_truths(udict, mapping_f):
    result = []
    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'Frame', 'Label', 'Preview URL']
    prefix = udict["prefix"]
    for pno in tqdm(udict["page"]):
        extract_required = lambda x: udict["page"][pno][x]
        required_keys = ['Text', 'ImageLoc', 'units']
        text, imgloc, units = list(map(extract_required, required_keys))
        image = cv2.imread(imgloc)
        rows, cols, _ = image.shape
        def trimRect(unit):
            rectKeys = ['rectLeft', 'rectTop', 'rectRight', 'rectBottom']
            x, y, X, Y = list(map(lambda x: int(unit[x]), rectKeys))
            #print(x, y, X, Y)
            trim_v = lambda w: lambda v: max(0, min(v, w))
            x = trim_v(cols)(x)
            X = trim_v(cols)(X)
            y = trim_v(rows)(y)
            Y = trim_v(rows)(Y)
            return x, y, X, Y
        # Order units by Key
        unit_truths, units = mapping_f(text, units)
        image_name = imgloc.split('/')[-1]

        for unit in units:
            x, y, X, Y = trimRect(unit)
            # cv2.rectangle(image, (x, y), (X, Y), (255,0,0), 3)
            result.append([x, y, X, Y, image_name, 'text', 'dummy'])
        # cv2.imwrite('temp/%s.jpg'%pno,image)
    path = '0002_Marthandavarma_Img_600_Original'
    df = pd.DataFrame(result, columns=columns)
    df.to_csv('%s/labels.csv'%path, index = False)


def read_book(**kwargs):
    book_dir_path = kwargs['book_path']
    opt_unit = kwargs['unit']
    obtainxml = lambda f: book_dir_path + f + '.xml'
    filenames = map(obtainxml, ['line', 'word', 'text'])
    lines, words, text = list(map(parse_ocr_xml, filenames))
    units = words
    mapping_f = word_mapping_f
    if opt_unit == 'line':
        units = lines
        mapping_f = line_mapping_f
    ud = group(text, units)
    ud["prefix"] = book_dir_path
    pagewise = images_and_truths(ud, mapping_f)
    
# give the desired book name
book = '/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0002/'
read_book(book_path=book, unit='line')