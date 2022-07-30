from selenium import webdriver
generated = [line.strip() for line in open("out_initial.txt", "r").readlines()]
relevant = [line.strip() for line in open("relevant_smiles.txt", "r").readlines()]
generated = list(set(generated))
temp = []
for smile in generated:
    if smile not in relevant:
        temp.append(smile)
generated = temp
valid = len(generated)
driver = webdriver.Chrome()
res_smiles = []
for smile in generated:
    driver.get("http://bioanalysis.cau.ac.kr:7030/")
    smiles_input = driver.find_element_by_name("sm")
    smiles_input.send_keys(smile)
    driver.find_element_by_css_selector("input[type='submit' i]").click()
    if driver.find_element_by_tag_name("td").text == "Permeable":
        res_smiles.append(smile)
    #if driver.find_element_by_tag_name("td").text == "Invalid SMILES":
    #    valid -= 1
#print(valid)
fout = open("out_processed.txt", "a")
for smile in res_smiles:
    fout.write(smile+"\n")
fout.close()