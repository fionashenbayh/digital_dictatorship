---
layout: page
title: Web Scraping
permalink: /tutorials/web_scrape_and_drive/
parent: Tutorials
nav_order: 1
---

```python
import requests, bs4, lxml, selenium
```


```python
response = requests.get('http://www.monitor.co.ug/News/National/Facebook-lawyer-Fred-Muwema/688334-3806268-mn7237/index.html')
```


```python
soup = bs4.BeautifulSoup(response.text, "lxml")
```


```python
title = soup.select('body > section.main-home.section > section > div > div > article > div > header > h1')[0].get_text()
```


```python
date = soup.select('body > section.main-home.section > section > div > div > article > div > header > h5')[0].get_text()
```


```python
summary = soup.select('body > section.main-home.section > section > div > div > article > div > article > section.article-sidebar > section.summary')[0].get_text()
```


```python
############################
# Scrape a single web page #
############################

import requests, bs4, lxml

# Website: UGANDAN DAILY MONITOR

# requests gets an HTML webpage
response = requests.get('http://www.monitor.co.ug/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html')

# BeautifulSoup parses HTML using special parser libraries, such as lxml
soup = bs4.BeautifulSoup(response.text, "lxml")

# Specific parts of the HTML document can be located using the CSS path


title = soup.select('')[0].get_text()
date = soup.select('')[0].get_text()
text = soup.select('')[0].get_text()

print(title, date, text)
```

    Moving on after losing hands and an eye to domestic violence 
    “I first got married to a bodaboda cyclist and we had two children. My husband got an accident and died and I was left alone to take care of the children who are now aged 10 and four. After sometime, I married another man, Adolf Busingye who is also a bodaboda cyclist.On the fateful day of  March 10, 2016, I was seven months pregnant, expecting his baby. A few days before that, we had developed a misunderstanding. I had cultivated maize and the harvest was promising. Busingye came back home and asked me to sell off the maize and give him the money because he wanted to purchase a motorcycle.My first husband had died due to a bodaboda accident on the road so when I married Busingye, who was a cyclist too, I was always worried about his life, and wanted to help him get another job. When he asked me for money to buy a motorcycle, I strongly objected. He did not like that and we quarreled.Despite the quarrel, I still felt that I stood a chance of winning the argument. I also thought he had been disturbed earlier, where he had spent the day and so I decided I should give him supper, sleep, and then talk to him the next day when he was sober. How it happened
    Faith Ninsiima breaks down as she narrates her ordeal. That did not happen. At around 8pm, I went to the kitchen to bring food. Busingye went to the store to bring a panga to mutilate me. As soon as I entered the house, even before I placed the food on the table, he cut off my left hand with a very sharp panga. I cried out but he moved forward and cut off my second hand. The panga was so sharp that he never had to cut twice! I turned to run but fell down and he also cut me on the head! This cost me my eye! He then ran away may be thinking I was dead. Getting helpIt took two hours for neighbours to come to my rescue. They took me to a nearby health facility where I was given First Aid  then I was transferred to Mulago hospital where I spent months nursing the wounds. My relatives and friends took care of me until I got better. While still bedridden, some NTV journalists approached me for an interview and it was through that broadcast that Justice Margaret Mutonyi, the Resident Judge of Mukono High Court came to know about me and visited me at Mulago. Back homeWhen I got better, I went back home but I was not my old self who used to do work on my own, take care of my children and also till our land for an income. My children had to be taken by friends since I could no longer cook or bathe them. I needed someone to look after me 24 hours. Life became really hard. However, I was so shocked that Busingye was still a free man eve after my relatives had reported the case to Kyenjojo Police Station. Up to now, he has never been arrested by police to face the law and he keeps on threatening my relatives telling them to withdraw the case. I fell very insecure whenever i think of going back to Kyenjojo as my attacker is loitering freely. A helping handOne day, Justice Mutonyi came home to check on me. She found me in that state and as a Good Samaritan, she decided to take me to her home to care for me, and also mobilise resources from her fellow female judges to get me artificial limbs so that I can do something for myself and children. I really thank God for her generosity. She has really taken good care of me and may God reward her abundantly.Greatest worries
    Ninsiima has now learnt how to use her phone although she still has a long way to go with bathing herself.However, I will never get married again basing on the tragic experience Mwesigye brought on me. My worry is about my three children and how they will achieve education. The eldest is Brian Atuhaire aged 10 now staying with his grandparents, David Atuyambe four, who stays with Pastor Ganyana at Banda Prayer Palace and my seven-month-old baby now in the care of Kyenjojo child centre.All I want is to have a business to make some money. If I can get artificial limbs, I shall be able to do something for myself and make money for my children. I feel bad that they are scattered among different people. I want them to be together. I wish I could stay with my children at least during the holidays so I can be a mother to them, bathe them and cook for them. I would also wish for now to get someone to help me take care of Brian my oldest child. He stays with his grandparents but they are very old; They are in their 80s.”Justice Margaret Mutonyi speaks about Ninsiima“The police have failed to produce Mwesigye before courts of the law yet the offence he committed is capital in nature. I visited her and exchanged contacts and after she was discharged I felt sorry for her and decided to help her from my home and also help her get artificial limbs so that she can do something for herself since she was a good enterprising young woman. 
    Ninsiima is now living in fear because her attacker is free and he continues to threaten her family.Since she came to my home, she has been cared for via clothing, feeding, bathing and everything but she keeps on worrying about her children who are not staying with her and who have been separated and given out to different people.The beauty about this young lady is that she is still hopeful and feels that she can do something to help support her three children who have currently been separated. The eldest who is 10 is still in Kyenjojo with her first late husband’s parents. The second who is four is the most affected of the three. He was adopted by Pastor Ganyana who says that when his friends try to beat him he always tells them that he will call Adolf who cut off his mother’s hands so that he can do the same to them. The baby is in a babies home back in Kyenjojo since she can’t take care of him.Ninsiima is not safe when the suspect is at large and that’s why she is taking refuge at my home. This is an extreme example of domestic violence which is irreversible and therefore justice should be done, police should arrest this man because his intention was to kill her.The cause of such domestic violence is as a result of failure to follow biblical principles; also poverty leads to domestic violence.”Meanwhile, the Rwenzori  Region spokesperson, Ms Lydia Tumushabe, said  in a telephone interview that they have not arrested Busingye  noting that he is on the run.   Saturday January 21 2017



```python
#######################################
# Scraping methods do not always work #
#######################################

# website: KENYAN DAILY NATION

response = requests.get('http://www.nation.co.ke/news/politics/Kibwana-joins-Kalonzo-s-Wiper-/1064-3806474-8wat13z/index.html')
soup = bs4.BeautifulSoup(response.text, "lxml")
```


```python
title = soup.select('body > article > section.main-home.section > section > div > div > article > div > header > h1')[0].get_text()
date = soup.select('body > article > section.main-home.section > section > div > div > article > div > header > h5')[0].get_text()
text = soup.select('body > article > section.main-home.section > section > div > div > article > div > article > section.body-copy')[0].get_text()
```


```python
# What's going on here?

print(soup.select('body > article > section.main-home.section > section > div > div > article > div > header > h1'))

#returns an empty list, nmeaning we are likely dealing with javascript, not html!
```

    []



```python
###########################################################
# Attempt Two, scraping websites programmed in JavaScript #
###########################################################

from selenium import webdriver 

#driver = webdriver.Firefox('specify/file/path/')  #use this command if you want to run the driver in Firefox
driver = webdriver.Chrome('/Users/Fiona_Shen_Bayh/Desktop/chromedriver') #I'm using Chrome

# Test it out on a single web page
driver.get('http://www.nation.co.ke/news/Moi-wanted-indepedence-delayed/1056-3790856-f1ol4pz/index.html')

#locate specific elements using the css path

title = driver.find_element_by_css_selector('body > article > section.main-home.section > section > div > div > article > div > header > h1')
date = driver.find_element_by_css_selector('body > article > section.main-home.section > section > div > div > article > div > header > h5')
main_text = driver.find_element_by_css_selector('body > article > section.main-home.section > section > div > div > article > div > article > section.body-copy')
# we don't have to use css here, we could also use driver.find_element_by_xpath()

#print the results

print(title.text)
print(date.text)
print(main_text.text)
driver.close()
```

    British intelligence documents show Moi wanted independence delayed
    SUNDAY JANUARY 29 2017
    By ODHIAMBO LEVIN OPIYO
    More by this Author
    Former President Daniel arap Moi was among leaders who had reservations about Kenya getting independence in 1963, recently released British intelligence documents show.
    Instead, he suggested that colonial rule be maintained for 10 years from 1959. Mr Moi’s conversations are contained in a declassified file boldly marked “secret” in red and titled “DT Arap Moi 1959”.
    It was at the height of agitation for freedom and African elected members in the Legislative Council (Legco) were closely being monitored by colonial security services in case they engaged in subversive activities.
    While touring the larger Nandi District in August 1959, and unaware that a close friend accompanying him was a British spy, Mr Moi is reported to have observed that granting Africans an early opportunity to govern themselves would not be in Kenya’s best interests.
    The document, written by the colonial director of intelligence and security and copied to Minister of Legal Affairs and Secretary to the Cabinet in Nairobi, further claims that Mr Moi’s thinking was that the Kenyan leadership should be given some level of responsibility around 1965 with independence following in 1970.
    Mr Moi, who was representing Rift Valley in the Legco, had just formed Kenya National Party (KNP)  in July 1959 with Mr Masinde Muliro, Mr Justus ole Tipis and Mr Ronald Ngala among others. 
    KNP positioned itself as a non-racial party, and included members of Legco of Asian and Arab origin.
    CHANGED THEIR MIND
    When the idea to form KNP was first mooted, other key politicians including Mr Jaramogi Oginga Odinga and Dr Julius Kiano had indicated they would be part of it but they later changed their mind to join Mr Tom Mboya in forming the Kenya Independence Movement (KIM).
    KNP, which was open to people of all races, demanded the delay in independence and opposed changes to the existing constitution.
    On the other hand KIM, whose membership was dominated by the Kikuyu and Luo communities, demanded changes to the Constitution, immediate independence and the release of Mzee Jomo Kenyatta. In 1960 after the first Lancaster conference KNP joined other parties in the formation of the Kenya African Democratic Union (Kadu) while KIM joined Kenya African Union (KAU) to form Kanu.
    Mr Moi would later sensationally claim that the decision by Mr Odinga, Mr Mboya and Mr Kiano not to join KNP was because the two Kikuyu and Luo politicians wanted to control the future government.
    He further reportedly argued that with such hegemony, opposition from smaller tribes would be difficult.
    While defending his decision to join KNP, Mr Moi had pointed out that in the Kalenjin areas in the Rift Valley, for example, there were relatively few Africans who were capable of taking over senior administration positions, meaning the dominant ethnic groups would occupy most top posts if independence was achieved early.
    Mr Moi, therefore, urged the Kalenjin to think seriously about education as top politicians, including in the Legco, appeared determined to ensure their ethnic communities gained an advantage.
    He gave the example of the famous student airlifts to America organised by Mr Mboya, which were apparently supposed to provide three “bursaries” to each constituency but ended up mostly benefiting people from the Kikuyu and Luo communities.
    ECONOMIC BOYCOTT
    However, a list of the students airlifted to study in the US in the 1950s and 1960s shows diversity, including people like Dorcas Boit who attended Spelman college and later became a director at the Kenya National Council of Social Services.
    According to the documents, Mr Moi further accused Mr Mboya of getting the support of the Asian community for African aspirations through threatening them with economic boycott.
    These attacks on Mr Mboya, according to the analysis by the colonial government’s Director of Intelligence Mr B.E. Wadeley, were well received by the Nandi who had started viewing Mr Moi as Mr Mboya’s “spanner boy”.
    “Moi’s moderate approach and his obvious sincerity created a good impression, and his criticism, albeit by inference of Mboya, was surprisingly well received,” Mr Wadeley observed.
    The analysis made by the colonial intelligence officials added: “At least he had identified himself as a Kalenjin and was not merely a mouthpiece of Tom Mboya and other extremists. His opposition to immediate self-government because of the danger of Kikuyu and Luo domination was well received by the Nandi.”
    The declassified documents claim that Mr Moi expressed his confidence that KNP would prevent any future domination of government posts by the “big” ethnic communities by slowing the progress towards independence and allowing the “small” tribes time to acquire education and position themselves for senior roles.
    Mr Moi also reportedly assured his supporters that apart from KNP he would form a Kalenjin political organisation, which also had the support of other leaders like Mr Masinde Muliro.
    SMALLER PARTIES
    The political organisation Mr Moi was talking about was the Kalenjin and Allies Central Governing Council (KACGC) – later renamed Kalenjin Political Alliance (KPA) – to largely protect the community’s interests, especially their land which was still under British occupation, from being taken over after independence.
    KPA lasted for only two months before joining other smaller parties to collapse into Kadu. Mr Moi’s tour of the larger Nandi District, according to the records, began on August 3, 1959, and ended on August 19, during which he held rallies at Kosirai attended by 500 people, Kipkaren trading centre (around 300 people), Chemundu location (700 people), Kaptumo trading centre (40 people), Cheptonge Moiben location (200 people), Chepkorier centre (60 people) and Sang’alo (300 people).
    Mr Moi was visiting the Nandi people after a long absence. 
    In the course of his tour he also held private meetings with some of his close friends, according to the declassified intelligence documents
    The irony of it all is that during his early political career Mr Moi championed devolution and a strong parliament contrary to his 24-year rule when he created a powerful presidency.
    On Saturday, Mr Moi’s long-time spokesman Lee Njiru told the Nation that “credible history” indicates the former president wanted independence immediately, adding that what is written by colonial historians should not always be believed.
    “He suffered discrimination as a teacher among colonial colleagues. He always tells us that whenever he expressed an opinion, they would tell him “you are not allowed to think. Just do what you are told.”
    And while agitating for freedom to access hotels reserved for whites, he was locked up in a cell and arraigned in court. He was charged alongside Jaramogi Oginga Odinga, Ronald Ngala and Masinde Muliro.
    Is it possible to suffer such indignities and at the same time seek to collude with the perpetrators? His visit to Mzee Kenyatta in detention shows he was for freedom,” he said.  
    “Mzee Moi used to feed and hide Mau Mau freedom fighters at his Nakuru home, this is documented in reputable history books.
    “Could he have done all these if he was not for Kenya’s attaining independence immediately? Treat some of these information put together by biased historians from the west with some pinch of salt.”
    email
    print
    ?
    
      by Taboola 
    Sponsored Links 
    You May Like
    Tiny Device Allows You To Track Anything (it's Genius!)
    Trackr Bravo
    This game will keep you up all night! Register for free!
    Forge Of Empires - Free Online Game
    This game will keep you up all night!
    Vikings: Free Online Game
    The New Travel Site That Just Kills it
    tripsinsider.com
    How this app teaches you a language in 3 weeks!
    Babbel
    World's longest aircraft gets off the ground
    Reuters TV
    A Solution That Puts Snoring to Bed
    My Snoring Solution
    The Silent Killer: Can Acid Reflux Become A Life Threatening Condition?
    Rapid Reflux Relief eBook
    In the headlines
    Bid to lower consent age to 16 dropped
    Duale writes to Speaker, withdrawing proposed amendments.
    Over 2 million don't wish to list as voters — poll
    12pc of Jubilee and 11pc of Cord/Nasa supporters plan to boycott August polls.
    Court blocks closure of Dadaab camps
    Garissa erupts in celebration at Farmajo win
    Police probe teachers filmed caning pupils
    Farmajo vows to rebuild Somalia - VIDEO
    Zuma deploys soldiers at parliament for annual address
    Former IEBC boss arrested over Chickengate scandal – VIDEO



```python
#####################################
# The web driver also works on HTML #
#####################################

#driver = webdriver.Firefox('specify/file/path/')
driver = webdriver.Chrome('/Users/Fiona_Shen_Bayh/Desktop/chromedriver')

# UGANDAN DAILY MONITOR
driver.get("http://www.monitor.co.ug/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html")

title = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h1')
date = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h5')
main_text = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/article/section[2]/div[1]')
print(title.text)
print(date.text)
print(main_text.text)
driver.close()
```

    Moving on after losing hands and an eye to domestic violence
    SATURDAY JANUARY 21 2017
    “I first got married to a bodaboda cyclist and we had two children. My husband got an accident and died and I was left alone to take care of the children who are now aged 10 and four. After sometime, I married another man, Adolf Busingye who is also a bodaboda cyclist.
    On the fateful day of March 10, 2016, I was seven months pregnant, expecting his baby. A few days before that, we had developed a misunderstanding. I had cultivated maize and the harvest was promising. Busingye came back home and asked me to sell off the maize and give him the money because he wanted to purchase a motorcycle.
    My first husband had died due to a bodaboda accident on the road so when I married Busingye, who was a cyclist too, I was always worried about his life, and wanted to help him get another job. When he asked me for money to buy a motorcycle, I strongly objected. He did not like that and we quarreled.
    Despite the quarrel, I still felt that I stood a chance of winning the argument. I also thought he had been disturbed earlier, where he had spent the day and so I decided I should give him supper, sleep, and then talk to him the next day when he was sober.
    How it happened
    Faith Ninsiima breaks down as she narrates her ordeal.
    
    That did not happen. At around 8pm, I went to the kitchen to bring food. Busingye went to the store to bring a panga to mutilate me. As soon as I entered the house, even before I placed the food on the table, he cut off my left hand with a very sharp panga.
    I cried out but he moved forward and cut off my second hand. The panga was so sharp that he never had to cut twice! I turned to run but fell down and he also cut me on the head! This cost me my eye! He then ran away may be thinking I was dead.
    Getting help
    It took two hours for neighbours to come to my rescue. They took me to a nearby health facility where I was given First Aid then I was transferred to Mulago hospital where I spent months nursing the wounds.
    My relatives and friends took care of me until I got better. While still bedridden, some NTV journalists approached me for an interview and it was through that broadcast that Justice Margaret Mutonyi, the Resident Judge of Mukono High Court came to know about me and visited me at Mulago.
    Back home
    When I got better, I went back home but I was not my old self who used to do work on my own, take care of my children and also till our land for an income. My children had to be taken by friends since I could no longer cook or bathe them.
    I needed someone to look after me 24 hours. Life became really hard. However, I was so shocked that Busingye was still a free man eve after my relatives had reported the case to Kyenjojo Police Station. Up to now, he has never been arrested by police to face the law and he keeps on threatening my relatives telling them to withdraw the case. I fell very insecure whenever i think of going back to Kyenjojo as my attacker is loitering freely.
    A helping hand
    One day, Justice Mutonyi came home to check on me. She found me in that state and as a Good Samaritan, she decided to take me to her home to care for me, and also mobilise resources from her fellow female judges to get me artificial limbs so that I can do something for myself and children. I really thank God for her generosity. She has really taken good care of me and may God reward her abundantly.
    Greatest worries
    Ninsiima has now learnt how to use her phone although she still has a long way to go with bathing herself.
    
    However, I will never get married again basing on the tragic experience Mwesigye brought on me. My worry is about my three children and how they will achieve education. The eldest is Brian Atuhaire aged 10 now staying with his grandparents, David Atuyambe four, who stays with Pastor Ganyana at Banda Prayer Palace and my seven-month-old baby now in the care of Kyenjojo child centre.
    All I want is to have a business to make some money. If I can get artificial limbs, I shall be able to do something for myself and make money for my children. I feel bad that they are scattered among different people. I want them to be together. I wish I could stay with my children at least during the holidays so I can be a mother to them, bathe them and cook for them. I would also wish for now to get someone to help me take care of Brian my oldest child. He stays with his grandparents but they are very old; They are in their 80s.”
    Justice Margaret Mutonyi speaks about Ninsiima
    “The police have failed to produce Mwesigye before courts of the law yet the offence he committed is capital in nature. I visited her and exchanged contacts and after she was discharged I felt sorry for her and decided to help her from my home and also help her get artificial limbs so that she can do something for herself since she was a good enterprising young woman.
    Ninsiima is now living in fear because her attacker is free and he continues to threaten her family.
    
    Since she came to my home, she has been cared for via clothing, feeding, bathing and everything but she keeps on worrying about her children who are not staying with her and who have been separated and given out to different people.



```python
##################################
# different ways to store output #
##################################

#note, you can only store information while the driver is still open!
driver = webdriver.Chrome('/Users/Fiona_Shen_Bayh/Desktop/chromedriver')
driver.get("http://www.monitor.co.ug/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html")
title = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h1')
date = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h5')
main_text = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/article/section[2]/div[1]')

#store output as a dictionary
article_dict = {}
article_dict['title'] = title.text
article_dict['date'] = date.text
article_dict['main_text'] = main_text.text

#store output as a list
article_list = [date.text, title.text, main_text.text]

#store output as a tuple
article_tup = (date.text, title.text, main_text.text)

driver.close()
```


```python
print(article_dict.items())
```

    dict_items([('main_text', '“I first got married to a bodaboda cyclist and we had two children. My husband got an accident and died and I was left alone to take care of the children who are now aged 10 and four. After sometime, I married another man, Adolf Busingye who is also a bodaboda cyclist.\nOn the fateful day of March 10, 2016, I was seven months pregnant, expecting his baby. A few days before that, we had developed a misunderstanding. I had cultivated maize and the harvest was promising. Busingye came back home and asked me to sell off the maize and give him the money because he wanted to purchase a motorcycle.\nMy first husband had died due to a bodaboda accident on the road so when I married Busingye, who was a cyclist too, I was always worried about his life, and wanted to help him get another job. When he asked me for money to buy a motorcycle, I strongly objected. He did not like that and we quarreled.\nDespite the quarrel, I still felt that I stood a chance of winning the argument. I also thought he had been disturbed earlier, where he had spent the day and so I decided I should give him supper, sleep, and then talk to him the next day when he was sober.\nHow it happened\nFaith Ninsiima breaks down as she narrates her ordeal.\n\nThat did not happen. At around 8pm, I went to the kitchen to bring food. Busingye went to the store to bring a panga to mutilate me. As soon as I entered the house, even before I placed the food on the table, he cut off my left hand with a very sharp panga.\nI cried out but he moved forward and cut off my second hand. The panga was so sharp that he never had to cut twice! I turned to run but fell down and he also cut me on the head! This cost me my eye! He then ran away may be thinking I was dead.\nGetting help\nIt took two hours for neighbours to come to my rescue. They took me to a nearby health facility where I was given First Aid then I was transferred to Mulago hospital where I spent months nursing the wounds.\nMy relatives and friends took care of me until I got better. While still bedridden, some NTV journalists approached me for an interview and it was through that broadcast that Justice Margaret Mutonyi, the Resident Judge of Mukono High Court came to know about me and visited me at Mulago.\nBack home\nWhen I got better, I went back home but I was not my old self who used to do work on my own, take care of my children and also till our land for an income. My children had to be taken by friends since I could no longer cook or bathe them.\nI needed someone to look after me 24 hours. Life became really hard. However, I was so shocked that Busingye was still a free man eve after my relatives had reported the case to Kyenjojo Police Station. Up to now, he has never been arrested by police to face the law and he keeps on threatening my relatives telling them to withdraw the case. I fell very insecure whenever i think of going back to Kyenjojo as my attacker is loitering freely.\nA helping hand\nOne day, Justice Mutonyi came home to check on me. She found me in that state and as a Good Samaritan, she decided to take me to her home to care for me, and also mobilise resources from her fellow female judges to get me artificial limbs so that I can do something for myself and children. I really thank God for her generosity. She has really taken good care of me and may God reward her abundantly.\nGreatest worries\nNinsiima has now learnt how to use her phone although she still has a long way to go with bathing herself.\n\nHowever, I will never get married again basing on the tragic experience Mwesigye brought on me. My worry is about my three children and how they will achieve education. The eldest is Brian Atuhaire aged 10 now staying with his grandparents, David Atuyambe four, who stays with Pastor Ganyana at Banda Prayer Palace and my seven-month-old baby now in the care of Kyenjojo child centre.\nAll I want is to have a business to make some money. If I can get artificial limbs, I shall be able to do something for myself and make money for my children. I feel bad that they are scattered among different people. I want them to be together. I wish I could stay with my children at least during the holidays so I can be a mother to them, bathe them and cook for them. I would also wish for now to get someone to help me take care of Brian my oldest child. He stays with his grandparents but they are very old; They are in their 80s.”\nJustice Margaret Mutonyi speaks about Ninsiima\n“The police have failed to produce Mwesigye before courts of the law yet the offence he committed is capital in nature. I visited her and exchanged contacts and after she was discharged I felt sorry for her and decided to help her from my home and also help her get artificial limbs so that she can do something for herself since she was a good enterprising young woman.\nNinsiima is now living in fear because her attacker is free and he continues to threaten her family.\n\nSince she came to my home, she has been cared for via clothing, feeding, bathing and everything but she keeps on worrying about her children who are not staying with her and who have been separated and given out to different people.'), ('title', 'Moving on after losing hands and an eye to domestic violence'), ('date', 'SATURDAY JANUARY 21 2017')])



```python
print(article_list)
```

    ['SATURDAY JANUARY 21 2017', 'Moving on after losing hands and an eye to domestic violence', '“I first got married to a bodaboda cyclist and we had two children. My husband got an accident and died and I was left alone to take care of the children who are now aged 10 and four. After sometime, I married another man, Adolf Busingye who is also a bodaboda cyclist.\nOn the fateful day of March 10, 2016, I was seven months pregnant, expecting his baby. A few days before that, we had developed a misunderstanding. I had cultivated maize and the harvest was promising. Busingye came back home and asked me to sell off the maize and give him the money because he wanted to purchase a motorcycle.\nMy first husband had died due to a bodaboda accident on the road so when I married Busingye, who was a cyclist too, I was always worried about his life, and wanted to help him get another job. When he asked me for money to buy a motorcycle, I strongly objected. He did not like that and we quarreled.\nDespite the quarrel, I still felt that I stood a chance of winning the argument. I also thought he had been disturbed earlier, where he had spent the day and so I decided I should give him supper, sleep, and then talk to him the next day when he was sober.\nHow it happened\nFaith Ninsiima breaks down as she narrates her ordeal.\n\nThat did not happen. At around 8pm, I went to the kitchen to bring food. Busingye went to the store to bring a panga to mutilate me. As soon as I entered the house, even before I placed the food on the table, he cut off my left hand with a very sharp panga.\nI cried out but he moved forward and cut off my second hand. The panga was so sharp that he never had to cut twice! I turned to run but fell down and he also cut me on the head! This cost me my eye! He then ran away may be thinking I was dead.\nGetting help\nIt took two hours for neighbours to come to my rescue. They took me to a nearby health facility where I was given First Aid then I was transferred to Mulago hospital where I spent months nursing the wounds.\nMy relatives and friends took care of me until I got better. While still bedridden, some NTV journalists approached me for an interview and it was through that broadcast that Justice Margaret Mutonyi, the Resident Judge of Mukono High Court came to know about me and visited me at Mulago.\nBack home\nWhen I got better, I went back home but I was not my old self who used to do work on my own, take care of my children and also till our land for an income. My children had to be taken by friends since I could no longer cook or bathe them.\nI needed someone to look after me 24 hours. Life became really hard. However, I was so shocked that Busingye was still a free man eve after my relatives had reported the case to Kyenjojo Police Station. Up to now, he has never been arrested by police to face the law and he keeps on threatening my relatives telling them to withdraw the case. I fell very insecure whenever i think of going back to Kyenjojo as my attacker is loitering freely.\nA helping hand\nOne day, Justice Mutonyi came home to check on me. She found me in that state and as a Good Samaritan, she decided to take me to her home to care for me, and also mobilise resources from her fellow female judges to get me artificial limbs so that I can do something for myself and children. I really thank God for her generosity. She has really taken good care of me and may God reward her abundantly.\nGreatest worries\nNinsiima has now learnt how to use her phone although she still has a long way to go with bathing herself.\n\nHowever, I will never get married again basing on the tragic experience Mwesigye brought on me. My worry is about my three children and how they will achieve education. The eldest is Brian Atuhaire aged 10 now staying with his grandparents, David Atuyambe four, who stays with Pastor Ganyana at Banda Prayer Palace and my seven-month-old baby now in the care of Kyenjojo child centre.\nAll I want is to have a business to make some money. If I can get artificial limbs, I shall be able to do something for myself and make money for my children. I feel bad that they are scattered among different people. I want them to be together. I wish I could stay with my children at least during the holidays so I can be a mother to them, bathe them and cook for them. I would also wish for now to get someone to help me take care of Brian my oldest child. He stays with his grandparents but they are very old; They are in their 80s.”\nJustice Margaret Mutonyi speaks about Ninsiima\n“The police have failed to produce Mwesigye before courts of the law yet the offence he committed is capital in nature. I visited her and exchanged contacts and after she was discharged I felt sorry for her and decided to help her from my home and also help her get artificial limbs so that she can do something for herself since she was a good enterprising young woman.\nNinsiima is now living in fear because her attacker is free and he continues to threaten her family.\n\nSince she came to my home, she has been cared for via clothing, feeding, bathing and everything but she keeps on worrying about her children who are not staying with her and who have been separated and given out to different people.']



```python
print(article_tup)
```

    ('SATURDAY JANUARY 21 2017', 'Moving on after losing hands and an eye to domestic violence', '“I first got married to a bodaboda cyclist and we had two children. My husband got an accident and died and I was left alone to take care of the children who are now aged 10 and four. After sometime, I married another man, Adolf Busingye who is also a bodaboda cyclist.\nOn the fateful day of March 10, 2016, I was seven months pregnant, expecting his baby. A few days before that, we had developed a misunderstanding. I had cultivated maize and the harvest was promising. Busingye came back home and asked me to sell off the maize and give him the money because he wanted to purchase a motorcycle.\nMy first husband had died due to a bodaboda accident on the road so when I married Busingye, who was a cyclist too, I was always worried about his life, and wanted to help him get another job. When he asked me for money to buy a motorcycle, I strongly objected. He did not like that and we quarreled.\nDespite the quarrel, I still felt that I stood a chance of winning the argument. I also thought he had been disturbed earlier, where he had spent the day and so I decided I should give him supper, sleep, and then talk to him the next day when he was sober.\nHow it happened\nFaith Ninsiima breaks down as she narrates her ordeal.\n\nThat did not happen. At around 8pm, I went to the kitchen to bring food. Busingye went to the store to bring a panga to mutilate me. As soon as I entered the house, even before I placed the food on the table, he cut off my left hand with a very sharp panga.\nI cried out but he moved forward and cut off my second hand. The panga was so sharp that he never had to cut twice! I turned to run but fell down and he also cut me on the head! This cost me my eye! He then ran away may be thinking I was dead.\nGetting help\nIt took two hours for neighbours to come to my rescue. They took me to a nearby health facility where I was given First Aid then I was transferred to Mulago hospital where I spent months nursing the wounds.\nMy relatives and friends took care of me until I got better. While still bedridden, some NTV journalists approached me for an interview and it was through that broadcast that Justice Margaret Mutonyi, the Resident Judge of Mukono High Court came to know about me and visited me at Mulago.\nBack home\nWhen I got better, I went back home but I was not my old self who used to do work on my own, take care of my children and also till our land for an income. My children had to be taken by friends since I could no longer cook or bathe them.\nI needed someone to look after me 24 hours. Life became really hard. However, I was so shocked that Busingye was still a free man eve after my relatives had reported the case to Kyenjojo Police Station. Up to now, he has never been arrested by police to face the law and he keeps on threatening my relatives telling them to withdraw the case. I fell very insecure whenever i think of going back to Kyenjojo as my attacker is loitering freely.\nA helping hand\nOne day, Justice Mutonyi came home to check on me. She found me in that state and as a Good Samaritan, she decided to take me to her home to care for me, and also mobilise resources from her fellow female judges to get me artificial limbs so that I can do something for myself and children. I really thank God for her generosity. She has really taken good care of me and may God reward her abundantly.\nGreatest worries\nNinsiima has now learnt how to use her phone although she still has a long way to go with bathing herself.\n\nHowever, I will never get married again basing on the tragic experience Mwesigye brought on me. My worry is about my three children and how they will achieve education. The eldest is Brian Atuhaire aged 10 now staying with his grandparents, David Atuyambe four, who stays with Pastor Ganyana at Banda Prayer Palace and my seven-month-old baby now in the care of Kyenjojo child centre.\nAll I want is to have a business to make some money. If I can get artificial limbs, I shall be able to do something for myself and make money for my children. I feel bad that they are scattered among different people. I want them to be together. I wish I could stay with my children at least during the holidays so I can be a mother to them, bathe them and cook for them. I would also wish for now to get someone to help me take care of Brian my oldest child. He stays with his grandparents but they are very old; They are in their 80s.”\nJustice Margaret Mutonyi speaks about Ninsiima\n“The police have failed to produce Mwesigye before courts of the law yet the offence he committed is capital in nature. I visited her and exchanged contacts and after she was discharged I felt sorry for her and decided to help her from my home and also help her get artificial limbs so that she can do something for herself since she was a good enterprising young woman.\nNinsiima is now living in fear because her attacker is free and he continues to threaten her family.\n\nSince she came to my home, she has been cared for via clothing, feeding, bathing and everything but she keeps on worrying about her children who are not staying with her and who have been separated and given out to different people.')



```python
import time # introduces pauses into our script
```


```python
base_url = 'http://www.monitor.co.ug/Magazines/'

url_pages = ['HealthLiving/689846-689846-vafaiv/index.html','PeoplePower/689844-689844-yyh79nz/index.html', 'Full-Woman/689842-689842-yra6q3z/index.html', 'Farming/689860-689860-yd72coz/index.html']
```


```python
all_the_links = []

for url in url_pages:
    time.sleep(1)
    response = requests.get(base_url + url)
    soup = bs4.BeautifulSoup(response.text, "lxml")
    links = [a.attrs.get('href') for a in soup.select()]

```


```python
###########################
# Scraping multiple links #
###########################

# For this particular site (the Ugandan Monitor), the main web page with all of the news article links can be scraped with requests and bs4
# we don't need to use selenium for this particular task

import time # introduces pauses into our script

#specify the base URL for the website and the specific URL pages
base_url = 'http://www.monitor.co.ug/Magazines'
url_page = '/PeoplePower/689844-689844-yyh79nz/index.html', '/Full-Woman/689842-689842-yra6q3z/index.html', '/Farming/689860-689860-yd72coz/index.html'

all_the_links = []
for url in url_page:
    time.sleep(1) # 1 second delay
    response = requests.get(base_url + url)
    soup = bs4.BeautifulSoup(response.text, "lxml")
    links = [a.attrs.get('href') for a in soup.select('body > section > article > section.section-home > div > div > section > div > div a[href^=/Magazines]')]
    all_the_links.extend(links)
```


```python
all_the_links
```




    ['/Magazines/PeoplePower/Why-wouldn-t-Busoga-celebrate-Gabula-job-/689844-3799390-7lm2vm/index.html',
     '/Magazines/PeoplePower/AU-motion-to-quit-ICC-left-in-limbo/689844-3799548-hqbu8g/index.html',
     '/Magazines/PeoplePower/Remembering-the-life-of-martyr-Janani-Luwum/689844-3799550-xmy635z/index.html',
     '/Magazines/PeoplePower/Is-Museveni-s-plan-to-wipe-out-opposition/689844-3799552-b5d4tvz/index.html',
     '/Magazines/PeoplePower/Understanding-evolution-of-financial-systems/689844-3799568-a1oeba/index.html',
     '/Magazines/PeoplePower/Ghana-s-Akufo-Addo-tips-Zuma-for-South-African-presidency/689844-3799560-w5sjo7z/index.html',
     '/Magazines/PeoplePower/Ghana-s-Akufo-Addo-tips-Zuma-for-South-African-presidency/689844-3799560-w5sjo7z/index.html',
     '/Magazines/PeoplePower/War-against-Museveni--UNLA-soldiers-kill-each-other/689844-3799564-12uyk4bz/index.html',
     '/Magazines/PeoplePower/War-against-Museveni--UNLA-soldiers-kill-each-other/689844-3799564-12uyk4bz/index.html',
     '/Magazines/PeoplePower/Demand-for-car-cash-returns--/689844-3799566-vktche/index.html',
     '/Magazines/PeoplePower/Demand-for-car-cash-returns--/689844-3799566-vktche/index.html',
     '/Magazines/PeoplePower/When-leaders-are-forced-out-of-power/689844-3790498-14t1fv6/index.html',
     '/Magazines/PeoplePower/When-leaders-are-forced-out-of-power/689844-3790498-14t1fv6/index.html',
     '/Magazines/PeoplePower/Uganda-supports-Kenya-s-Amina-Mohamed-for-AU-job/689844-3790512-gl5hea/index.html',
     '/Magazines/PeoplePower/Uganda-supports-Kenya-s-Amina-Mohamed-for-AU-job/689844-3790512-gl5hea/index.html',
     '/Magazines/PeoplePower/-Africa--must-plan-for-the-growing-population-/689844-3790514-laak0hz/index.html',
     '/Magazines/PeoplePower/-Africa--must-plan-for-the-growing-population-/689844-3790514-laak0hz/index.html',
     '/Magazines/PeoplePower/Col-Kaka--the-new-spy-chief-at-ISO/689844-3790516-2qsfs6z/index.html',
     '/Magazines/PeoplePower/Col-Kaka--the-new-spy-chief-at-ISO/689844-3790516-2qsfs6z/index.html',
     '/Magazines/Full-Woman/A-thankful-heart-is-medicine-in-itself/689842-3798478-336jxuz/index.html',
     '/Magazines/Full-Woman/She-lives-her-passion-stitch-by-stitch/689842-3798404-91e8cnz/index.html',
     '/Magazines/Full-Woman/Since-when-did-social-media-become-an-authority-on-our-lives-/689842-3798420-103o38cz/index.html',
     '/Magazines/Full-Woman/She-has-found-her-niche-in-wedding-planning/689842-3798468-pee1vjz/index.html',
     '/Magazines/Full-Woman/Why-children-are-disrespectful-nowadays-and-how-to-deal-with-it/689842-3798488-ykillez/index.html',
     '/Magazines/Full-Woman/My-music-is-inspired-by-my-relationship/689842-3798496-h19k36z/index.html',
     '/Magazines/Full-Woman/My-music-is-inspired-by-my-relationship/689842-3798496-h19k36z/index.html',
     '/Magazines/Full-Woman/Treading-the-path-of-motherhood-and-career/689842-3798458-b0rudw/index.html',
     '/Magazines/Full-Woman/Treading-the-path-of-motherhood-and-career/689842-3798458-b0rudw/index.html',
     '/Magazines/Full-Woman/I-wanted-to-be-a-priest--Robert-Bake-Tumuhaise/689842-3791568-u7kmpcz/index.html',
     '/Magazines/Full-Woman/I-wanted-to-be-a-priest--Robert-Bake-Tumuhaise/689842-3791568-u7kmpcz/index.html',
     '/Magazines/Full-Woman/It-takes-time-for-one-to-master-something/689842-3791566-p08hbvz/index.html',
     '/Magazines/Full-Woman/It-takes-time-for-one-to-master-something/689842-3791566-p08hbvz/index.html',
     '/Magazines/Full-Woman/From-waitress-to-club-manager/689842-3791558-mnxjyl/index.html',
     '/Magazines/Full-Woman/From-waitress-to-club-manager/689842-3791558-mnxjyl/index.html',
     '/Magazines/Full-Woman/Never-lose-your-feminine-instinct--Sophie-Ikenye/689842-3791552-14mtft1/index.html',
     '/Magazines/Full-Woman/Never-lose-your-feminine-instinct--Sophie-Ikenye/689842-3791552-14mtft1/index.html',
     '/Magazines/Full-Woman/I-wouldn-t-want-to-be-in-Melania-s-shoes/689842-3791532-yl0wpyz/index.html',
     '/Magazines/Full-Woman/I-wouldn-t-want-to-be-in-Melania-s-shoes/689842-3791532-yl0wpyz/index.html',
     '/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html',
     '/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html',
     '/Magazines/Full-Woman/Where-is-the-rain-/689842-3780838-9t399q/index.html',
     '/Magazines/Full-Woman/Where-is-the-rain-/689842-3780838-9t399q/index.html',
     '/Magazines/Full-Woman/Young-girls-are-not-for-me/689842-3780892-fye1et/index.html',
     '/Magazines/Full-Woman/Young-girls-are-not-for-me/689842-3780892-fye1et/index.html',
     '/Magazines/Full-Woman/Earning-his-pocket-money/689842-3780912-w4677b/index.html',
     '/Magazines/Full-Woman/Earning-his-pocket-money/689842-3780912-w4677b/index.html',
     '/Magazines/Full-Woman/I-am-single-and-searching/689842-3780920-nduj4qz/index.html',
     '/Magazines/Full-Woman/I-am-single-and-searching/689842-3780920-nduj4qz/index.html',
     '/Magazines/Full-Woman/Woman-helm-Uganda-Institution-Professional-Engineers/689842-3780876-tpt72jz/index.html',
     '/Magazines/Full-Woman/Woman-helm-Uganda-Institution-Professional-Engineers/689842-3780876-tpt72jz/index.html',
     '/Magazines/Farming/USAID-post-harvest-technologies-IREN/689860-3801882-osktfjz/index.html',
     '/Magazines/Farming/What-do-I-need-to-establish-a-commercial-fish-farming-business-/689860-3799708-10e1g2p/index.html',
     '/Magazines/Farming/Propolis--The-bee-product-on-demand/689860-3799686-106g5dr/index.html',
     '/Magazines/Farming/Farmers-and-uncertain-weather/689860-3799696-q6n15tz/index.html',
     '/Magazines/Farming/How-we-can-get-more-from-the-banana-stem/689860-3799674-12fcaw1/index.html',
     '/Magazines/Farming/Refugees--drought-affect-food-supply-in-Arua-markets/689860-3799680-45u6rm/index.html',
     '/Magazines/Farming/Refugees--drought-affect-food-supply-in-Arua-markets/689860-3799680-45u6rm/index.html',
     '/Magazines/Farming/Making-mixed-farming-work-in-a-small-space/689860-3799662-g361hs/index.html',
     '/Magazines/Farming/Making-mixed-farming-work-in-a-small-space/689860-3799662-g361hs/index.html',
     '/Magazines/Farming/-value-Food-Control-agricultural-sector--pulp--paper/689860-3789336-elf86kz/index.html',
     '/Magazines/Farming/-value-Food-Control-agricultural-sector--pulp--paper/689860-3789336-elf86kz/index.html',
     '/Magazines/Farming/Rice-grow---demand-food-Iganga-Hoima--product-/689860-3789358-uu5e72/index.html',
     '/Magazines/Farming/Rice-grow---demand-food-Iganga-Hoima--product-/689860-3789358-uu5e72/index.html',
     '/Magazines/Farming/Trees--life--Eucalyptus-plants/689860-3789326-122pgvyz/index.html',
     '/Magazines/Farming/Trees--life--Eucalyptus-plants/689860-3789326-122pgvyz/index.html',
     '/Magazines/Farming/Bees--source-products--Venom-honey--propolis/689860-3789402-k6m44v/index.html',
     '/Magazines/Farming/Bees--source-products--Venom-honey--propolis/689860-3789402-k6m44v/index.html',
     '/Magazines/Farming/Guard--livestock--diseases-farmers/689860-3789366-3gno6h/index.html',
     '/Magazines/Farming/Guard--livestock--diseases-farmers/689860-3789366-3gno6h/index.html',
     '/Magazines/Farming/-hybrid-bananas--matooke-East-African-Highland-NARO/689860-3789350-su6qpk/index.html',
     '/Magazines/Farming/-hybrid-bananas--matooke-East-African-Highland-NARO/689860-3789350-su6qpk/index.html',
     '/Magazines/Farming/Farmers---insurance--scheme--drought/689860-3789374-3tqin5/index.html',
     '/Magazines/Farming/Farmers---insurance--scheme--drought/689860-3789374-3tqin5/index.html',
     '/Magazines/Farming/manage--livestock---droughts--veterinarians-scientists/689860-3789382-w87dfez/index.html',
     '/Magazines/Farming/manage--livestock---droughts--veterinarians-scientists/689860-3789382-w87dfez/index.html',
     '/Magazines/Farming/What-is-initial-capital-to-start-a-fish-farming-business-/689860-3781626-4ypg1vz/index.html',
     '/Magazines/Farming/What-is-initial-capital-to-start-a-fish-farming-business-/689860-3781626-4ypg1vz/index.html',
     '/Magazines/Farming/Avian-flu--What-farmers-should-do/689860-3781614-fn0d74/index.html',
     '/Magazines/Farming/Avian-flu--What-farmers-should-do/689860-3781614-fn0d74/index.html',
     '/Magazines/Farming/Long-dry-season--schools-re-opening-affects-bean-prices/689860-3781618-dmv92cz/index.html',
     '/Magazines/Farming/Long-dry-season--schools-re-opening-affects-bean-prices/689860-3781618-dmv92cz/index.html']




```python
all_the_links = set(all_the_links) #remove duplicates
print(all_the_links)
```

    {'/Magazines/Full-Woman/A-thankful-heart-is-medicine-in-itself/689842-3798478-336jxuz/index.html', '/Magazines/PeoplePower/Remembering-the-life-of-martyr-Janani-Luwum/689844-3799550-xmy635z/index.html', '/Magazines/Full-Woman/Woman-helm-Uganda-Institution-Professional-Engineers/689842-3780876-tpt72jz/index.html', '/Magazines/Farming/What-do-I-need-to-establish-a-commercial-fish-farming-business-/689860-3799708-10e1g2p/index.html', '/Magazines/Farming/Propolis--The-bee-product-on-demand/689860-3799686-106g5dr/index.html', '/Magazines/Farming/What-is-initial-capital-to-start-a-fish-farming-business-/689860-3781626-4ypg1vz/index.html', '/Magazines/Farming/-hybrid-bananas--matooke-East-African-Highland-NARO/689860-3789350-su6qpk/index.html', '/Magazines/Full-Woman/Why-children-are-disrespectful-nowadays-and-how-to-deal-with-it/689842-3798488-ykillez/index.html', '/Magazines/Full-Woman/It-takes-time-for-one-to-master-something/689842-3791566-p08hbvz/index.html', '/Magazines/Full-Woman/I-wanted-to-be-a-priest--Robert-Bake-Tumuhaise/689842-3791568-u7kmpcz/index.html', '/Magazines/Full-Woman/I-wouldn-t-want-to-be-in-Melania-s-shoes/689842-3791532-yl0wpyz/index.html', '/Magazines/Farming/How-we-can-get-more-from-the-banana-stem/689860-3799674-12fcaw1/index.html', '/Magazines/Farming/Farmers---insurance--scheme--drought/689860-3789374-3tqin5/index.html', '/Magazines/Full-Woman/My-music-is-inspired-by-my-relationship/689842-3798496-h19k36z/index.html', '/Magazines/Farming/Making-mixed-farming-work-in-a-small-space/689860-3799662-g361hs/index.html', '/Magazines/Full-Woman/Never-lose-your-feminine-instinct--Sophie-Ikenye/689842-3791552-14mtft1/index.html', '/Magazines/PeoplePower/Ghana-s-Akufo-Addo-tips-Zuma-for-South-African-presidency/689844-3799560-w5sjo7z/index.html', '/Magazines/PeoplePower/Understanding-evolution-of-financial-systems/689844-3799568-a1oeba/index.html', '/Magazines/Farming/Trees--life--Eucalyptus-plants/689860-3789326-122pgvyz/index.html', '/Magazines/Full-Woman/Moving-on-after-losing-hands-and-an-eye-to-domestic-violence/689842-3780864-t0ba4z/index.html', '/Magazines/PeoplePower/War-against-Museveni--UNLA-soldiers-kill-each-other/689844-3799564-12uyk4bz/index.html', '/Magazines/PeoplePower/Why-wouldn-t-Busoga-celebrate-Gabula-job-/689844-3799390-7lm2vm/index.html', '/Magazines/PeoplePower/Is-Museveni-s-plan-to-wipe-out-opposition/689844-3799552-b5d4tvz/index.html', '/Magazines/Farming/USAID-post-harvest-technologies-IREN/689860-3801882-osktfjz/index.html', '/Magazines/Full-Woman/She-has-found-her-niche-in-wedding-planning/689842-3798468-pee1vjz/index.html', '/Magazines/Full-Woman/Young-girls-are-not-for-me/689842-3780892-fye1et/index.html', '/Magazines/Farming/Refugees--drought-affect-food-supply-in-Arua-markets/689860-3799680-45u6rm/index.html', '/Magazines/Full-Woman/From-waitress-to-club-manager/689842-3791558-mnxjyl/index.html', '/Magazines/Farming/Guard--livestock--diseases-farmers/689860-3789366-3gno6h/index.html', '/Magazines/Full-Woman/Where-is-the-rain-/689842-3780838-9t399q/index.html', '/Magazines/PeoplePower/Uganda-supports-Kenya-s-Amina-Mohamed-for-AU-job/689844-3790512-gl5hea/index.html', '/Magazines/Full-Woman/She-lives-her-passion-stitch-by-stitch/689842-3798404-91e8cnz/index.html', '/Magazines/Full-Woman/Treading-the-path-of-motherhood-and-career/689842-3798458-b0rudw/index.html', '/Magazines/Farming/Avian-flu--What-farmers-should-do/689860-3781614-fn0d74/index.html', '/Magazines/Farming/Long-dry-season--schools-re-opening-affects-bean-prices/689860-3781618-dmv92cz/index.html', '/Magazines/Full-Woman/I-am-single-and-searching/689842-3780920-nduj4qz/index.html', '/Magazines/Farming/Bees--source-products--Venom-honey--propolis/689860-3789402-k6m44v/index.html', '/Magazines/Farming/manage--livestock---droughts--veterinarians-scientists/689860-3789382-w87dfez/index.html', '/Magazines/Farming/Farmers-and-uncertain-weather/689860-3799696-q6n15tz/index.html', '/Magazines/PeoplePower/Col-Kaka--the-new-spy-chief-at-ISO/689844-3790516-2qsfs6z/index.html', '/Magazines/Full-Woman/Earning-his-pocket-money/689842-3780912-w4677b/index.html', '/Magazines/PeoplePower/Demand-for-car-cash-returns--/689844-3799566-vktche/index.html', '/Magazines/PeoplePower/-Africa--must-plan-for-the-growing-population-/689844-3790514-laak0hz/index.html', '/Magazines/Farming/Rice-grow---demand-food-Iganga-Hoima--product-/689860-3789358-uu5e72/index.html', '/Magazines/Full-Woman/Since-when-did-social-media-become-an-authority-on-our-lives-/689842-3798420-103o38cz/index.html', '/Magazines/PeoplePower/When-leaders-are-forced-out-of-power/689844-3790498-14t1fv6/index.html', '/Magazines/Farming/-value-Food-Control-agricultural-sector--pulp--paper/689860-3789336-elf86kz/index.html', '/Magazines/PeoplePower/AU-motion-to-quit-ICC-left-in-limbo/689844-3799548-hqbu8g/index.html'}



```python
base_url = 'http://www.monitor.co.ug'

all_the_articles = [] # create an empty list to store article content

import time 
# this inserts pauses into our script, which may be necessary when scraping certain sites
# sometimes websites will crash or deny access if they receive too many requests too quickly

for link in all_the_links:
    time.sleep(1) # pause 1 second between every driver instance
    mini_dict = {} #create an empty dictionary to store each set of results
    driver = webdriver.Chrome('/Users/Fiona_Shen_Bayh/Desktop/chromedriver')
    driver.get(base_url + link)
    title = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h1')
    date = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h5')
    main_text = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/article/section[2]/div[1]')
    mini_dict['title'] = title.text
    mini_dict['date'] = date.text
    mini_dict['main_text'] = main_text.text
    all_the_articles.append(mini_dict) #add each set of results to the main list
    driver.close() #close the driver after each instance
  
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-40-e832f6bdf6e6> in <module>()
         11     mini_dict = {} #create an empty dictionary to store each set of results
         12     driver = webdriver.Chrome('/Users/Fiona_Shen_Bayh/Desktop/chromedriver')
    ---> 13     driver.get(base_url + link)
         14     title = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h1')
         15     date = driver.find_element_by_xpath('/html/body/section[2]/section/div/div/article/div/header/h5')


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/site-packages/selenium/webdriver/remote/webdriver.py in get(self, url)
        246         Loads a web page in the current browser session.
        247         """
    --> 248         self.execute(Command.GET, {'url': url})
        249 
        250     @property


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/site-packages/selenium/webdriver/remote/webdriver.py in execute(self, driver_command, params)
        232 
        233         params = self._wrap_value(params)
    --> 234         response = self.command_executor.execute(driver_command, params)
        235         if response:
        236             self.error_handler.check_response(response)


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/site-packages/selenium/webdriver/remote/remote_connection.py in execute(self, command, params)
        406         path = string.Template(command_info[1]).substitute(params)
        407         url = '%s%s' % (self._url, path)
    --> 408         return self._request(command_info[0], url, body=data)
        409 
        410     def _request(self, method, url, body=None):


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/site-packages/selenium/webdriver/remote/remote_connection.py in _request(self, method, url, body)
        438             try:
        439                 self._conn.request(method, parsed_url.path, body, headers)
    --> 440                 resp = self._conn.getresponse()
        441             except (httplib.HTTPException, socket.error):
        442                 self._conn.close()


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/http/client.py in getresponse(self)
       1195         try:
       1196             try:
    -> 1197                 response.begin()
       1198             except ConnectionError:
       1199                 self.close()


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/http/client.py in begin(self)
        295         # read until we get a non-100 response
        296         while True:
    --> 297             version, status, reason = self._read_status()
        298             if status != CONTINUE:
        299                 break


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/http/client.py in _read_status(self)
        256 
        257     def _read_status(self):
    --> 258         line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        259         if len(line) > _MAXLINE:
        260             raise LineTooLong("status line")


    /Users/Fiona_Shen_Bayh/anaconda/lib/python3.5/socket.py in readinto(self, b)
        573         while True:
        574             try:
    --> 575                 return self._sock.recv_into(b)
        576             except timeout:
        577                 self._timeout_occurred = True


    KeyboardInterrupt: 



```python
all_the_articles[0] #check the first entry in our list
```




    {'date': 'SUNDAY FEBRUARY 5 2017',
     'main_text': 'Before the year started, we set out with what I termed as our long tin of thankfulness. I bought little coloured notes to help us. Each day starting January 1, each of us would write a note with a date of thankfulness to God for one thing or the other. The idea was for us to focus on a heart of thanksgiving as opposed to grumbling and thinking alot about the many negative things that surround us.\nThere are many things to be sad about, those regularly present on a moment by moment basis. From the sibling who switches on the light very early in the morning to the one who didn’t leave the bathroom in a desired state, to one who bangs doors and the one who wasn’t polite.\nFrom the erratic drivers on the road who create several lanes and cause an unnecessary traffic jam. Everyone seems to be so angry one leader said he isn’t a servant. If it’s not a boda boda slumming your recently repaired second hand new Japanese car, it’s the potholes or the excessive heat wave that has hit and reminded us how we have destroyed our environment.\nThe many things to complain about surround and drain our energy and thus the choice to have the tin of ‘thank you’ notes.\nThe trick is in consistency, a little similar to new year resolutions. I placed the beautiful transparent tin in a place that would force us to see it and use it daily, the toilet area. Next to it was a beautiful set of pens along with the colour notes. A week or two into the project, the children had given up. So I got into a discussion with one of them so that we can reflect on the power of coming through on one’s word.\nThere is no strength, no gain in committing on a thing and failing to complete it. Mastery is the art and ability to do something beyond excellently even when it’s inconvenient. The ability to be consistent. So we committed on a 12months project and here we are, barely a month into it and we have forgotten about it. How can we claim to be persons of excellence when we cannot carry through the seemingly simple task of having something to be thankful for each day for 365 days?\nI am thankful for cool weather, thankful for a nice warm bed, a hot shower in the evening, thankful for a friend or two with whom I can have meaningful conversation. I am thankful for health, thankful for the neighbour who hoots very loudly and reminds me that I actually have ears that work. I am thankful for a time like this when I can share life lessons and skills with the children, for example, the gift of thankfulness. There’s indeed so much to frown on but a thankful heart is medicine in itself. I am thankful for regular meals and the ability to eat and drink. I thank God for life and a chance to remember to be thankful.\n-jmabola@yahoo.com',
     'title': 'A thankful heart is medicine in itself'}


