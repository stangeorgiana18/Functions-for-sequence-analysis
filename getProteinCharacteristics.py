import urllib3
import re
import requests
from urllib.request import urlopen


# get protein sequence:

link = "http://rest.uniprot.org/uniprotkb/P04637.fasta" 
http = urllib3.PoolManager()
f = http.request("GET",link)
print (f.data)



# protein characteristics for a string with proteins

def getProteinCharacteristics(allProteins, proteinType):
    def NumberofAminoacids(Protein):
        lines = [line.strip() for line in Protein.split("\n")[1:]]
        aminoacidCount = ""
        for line in lines:
            aminoacidCount += line
        return len(aminoacidCount)
    
    # split the Proteins string by '>tr'
    splitProteins =  allProteins.split(">tr")[1:]
  
    result = [
        {
            "proteinName": Protein.split("|")[1],
            "numberOfAminoacids": NumberofAminoacids(Protein),
            "proteinType": proteinType,
        }
        for Protein in splitProteins
    ]

    return result


allProteins = """>tr|A0A024QZ26|A0A024QZ26_HUMAN Histone deacetylase 6, isoform CRA_c OS=Homo sapiens OX=9606 GN=HDAC6 PE=3 SV=1 
MTSTGQDSTTTRQRRSRQNPQSPPQDSSVTSKRNIKKGAVPRSIPNLAEVKKKGKMKKLG
QAMEEDLIVGLQGMDLNLEAEALAGTGLVLDEQLNEFHCLWDDSFPEGPERLHAIKEQLI
QEGLLDRCVSFQARFAEKEELMLVHSLEYIDLMETTQYMNEGELRVLADTYDSVYLHPNS
YSCACLASGSVLRLVDAVLGAEIRNGMAIIRPPGHHAQHSLMDGYCMFNHVAVAARYAQQ
KHRIRRVLIVDWDVHHGQGTQFTFDQDPSVLYFSIHRYEQGRFWPHLKASNWSTTGFGQG
QGYTINVPWNQVGMRDADYIAAFLHVLLPVALEFQPQLVLVAAGFDALQGDPKGEMAATP
AGFAQLTHLLMGLAGGKLILSLEGGYNLRALAEGVSASLHTLLGDPCPMLESPGAPCRSA
QASVSCALEALEPFWEVLVRSTETVERDNMEEDNVEESEEEGPWEPPVLPILTWPVLQSR
TGLVYDQNMMNHCNLWDSHHPEVPQRILRIMCRLEELGLAGRCLTLTPRPATEAELLTCH
SAEYVGHLRATEKMKTRELHRESSNFDSIYICPSTFACAQLATGAACRLVEAVLSGEVLN
GAAVVRPPGHHAEQDAACGFCFFNSVAVAARHAQTISGHALRILIVDWDVHHGNGTQHMF
EDDPSVLYVSLHRYDHGTFFPMGDEGASSQIGRAAGTGFTVNVAWNGPRMGDADYLAAWH
RLVLPIAYEFNPELVLVSAGFDAARGDPLGGCQVSPEGYAHLTHLLMGLASGRIILILEG
GYNLTSISESMAACTRSLLGDPPPLLTLPRPPLSGALASITETIQVHRRYWRSLRVMKVE
DREGPSSSKLVTKKAPQPAKPRLAERMTTREKKVLEAGMGKVTSASFGEESTPGQTNSET
AVVALTQDQPSEAATGGATLAQTISEAAIGGAMLGQTTSEEAVGGATPDQTTSEETVGGA
ILDQTTSEDAVGGATLGQTTSEEAVGGATLAQTTSEAAMEGATLDQTTSEEAPGGTELIQ
TPLASSTDHQTPPTSPVQGTTPQISPSTLIGSLRTLELGSESQGASESQAPGEENLLGEA
AGGQDMADSMLMQGSRGLTDQAIFYAVTPLPWCPHLVAVCPIPAAGLDVTQPCGDCGTIQ
ENWVCLSCYQVYCGRYINGHMLQHHGNSGHPLVLSYIDLSAWCYYCQAYVHHQALLDVKN
IAHQNKFGEDMPHPH
>tr|A0A024QZ96|A0A024QZ96_HUMAN T-box 2, isoform CRA_a OS=Homo sapiens OX=9606 GN=TBX2 PE=4 SV=1
MREPALAASAMAYHPFHAPRPADFPMSAFLAAAQPSFFPALALPPGALAKPLPDPGLAGA
AAAAAAAAAAAEAGLHVSALGPHPPAAHLRSLKSLEPEDEVEDDPKVTLEAKELWDQFHK
LGTEMVITKSGRRMFPPFKVRVSGLDKKAKYILLMDIVAADDCRYKFHNSRWMVAGKADP
EMPKRMYIHPDSPATGEQWMAKPVAFHKLKLTNNISDKHGFTILNSMHKYQPRFHIVRAN
DILKLPYSTFRTYVFPETDFIAVTAYQNDKITQLKIDNNPFAKGFRDTGNGRREKRKQLT
LPSLRLYEEHCKPERDGAESDASSCDPPPAREPPTSPGAAPSPLRLHRARAEEKSCAADS
DPEPERLSEERAGAPLGRSPAPDSASPTRLTEPERARERRSPERGKEPAESGGDGPFGLR
SLEKERAEARRKDEGRKEAAEGKEQGLAPLVVQTDSASPLGAGHLPGLAFSSHLHGQQFF
GPLGAGQPLFLHPGQFTMGPGAFSAMGMGHLLASVAGGGNGGGGGPGTAAGLDAGGLGPA
ASAASTAAPFPFHLSQHMLASQGIPMPTFGGLFPYPYTYMAAAAAAASALPATSAAAAAA
AAAGSLSRSPFLGSARPRLRFSPYQIPVTIPPSTSLLTTGLASEGSKAAGGNSREPSPLP
ELALRKVGAPSRGALSPSGSAKEAANELQSIQRLVSGLESQRALSPGRESPK
>tr|A0A024QZA8|A0A024QZA8_HUMAN receptor protein-tyrosine kinase OS=Homo sapiens OX=9606 GN=EPHA2 PE=4 SV=1
MELQAARACFALLWGCALAAAAAAQGKEVVLLDFAAAGGELGWLTHPYGKGWDLMQNIMN
DMPIYMYSVCNVMSGDQDNWLRTNWVYRGEAERIFIELKFTVRDCNSFPGGASSCKETFN
LYYAESDLDYGTNFQKRLFTKIDTIAPDEITVSSDFEARHVKLNVEERSVGPLTRKGFYL
AFQDIGACVALLSVRVYYKKCPELLQGLAHFPETIAGSDAPSLATVAGTCVDHAVVPPGG
EEPRMHCAVDGEWLVPIGQCLCQAGYEKVEDACQACSPGFFKFEASESPCLECPEHTLPS
PEGATSCECEEGFFRAPQDPASMPCTRPPSAPHYLTAVGMGAKVELRWTPPQDSGGREDI
VYSVTCEQCWPESGECGPCEASVRYSEPPHGLTRTSVTVSDLEPHMNYTFTVEARNGVSG
LVTSRSFRTASVSINQTEPPKVRLEGRSTTSLSVSWSIPPPQQSRVWKYEVTYRKKGDSN
SYNVRRTEGFSVTLDDLAPDTTYLVQVQALTQEGQGAGSKVHEFQTLSPEGSGNLAVIGG
VAVGVVLLLVLAGVGFFIHRRRKNQRARQSPEDVYFSKSEQLKPLKTYVDPHTYEDPNQA
VLKFTTEIHPSCVTRQKVIGAGEFGEVYKGMLKTSSGKKEVPVAIKTLKAGYTEKQRVDF
LGEAGIMGQFSHHNIIRLEGVISKYKPMMIITEYMENGALDKFLREKDGEFSVLQLVGML
RGIAAGMKYLANMNYVHRDLAARNILVNSNLVCKVSDFGLSRVLEDDPEATYTTSGGKIP
IRWTAPEAISYRKFTSASDVWSFGIVMWEVMTYGERPYWELSNHEVMKAINDGFRLPTPM
DCPSAIYQLMMQCWQQERARRPKFADIVSILDKLIRAPDSLKTLADFDPRVSIRLPSTSG
SEGVPFRTVSEWLESIKMQQYTEHFMAAGYTAIEKVVQMTNDDIKRIGVRLPGHQKRIAY
SLLGLKDQVNTVGIPI"""

proteinCharacteristics = getProteinCharacteristics(allProteins, 1)
print(proteinCharacteristics)
