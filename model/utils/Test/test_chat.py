try:  # Local import to avoid circular dependency when running unit tests.
    from model.utils.proxy_call import OpenaiCall
    from model.utils.all_en_prompts import (
        get_poi_description_prompt,
        get_poi_extraction_prompt,
        get_poi_extraction_prompt_new
    )
except ImportError:  # pragma: no cover - handled during runtime.
    OpenaiCall = None  # type: ignore


text = "上海文艺指南 WANGR 古籍井店 佳善理下 ONOS Charaluo 0 A区 黄浦区 静安 TEF 大 CH GNING 云门 长宁区 Shonghal 兴国 BARU ROND Heraius ！ 国 徐汇区 大美术 电物 GULMROAD 西岸面 NO-A UNUINRUAD TEMIPE ROA 上海文 艺指南 外滩美术馆 HAINING RU 古籍书店 BUDIHA 慈善超市 TEMPL 佛禅 浦东美术馆 ChairClub 汇丰纸行 HUANGPU JING'AN Shal 黄浦区 TELE candore'sSqiare TC 静安区 东 育音堂音乐公园 每 上海大剧院 linganTem CHAO IGNING 云峰剧院 寺 长宁区 塞万 复兴公园 花园饭店 提斯 Shanghai DROAD Culure Square 思南公馆 当代艺术 兴国宾馆 抓鱼 夕 文响音乐 博物馆 博物馆 DAPU ROAD Papermoon Shan 上剧场 Jiaotong Herlarious D SHAN Universit Xuhui Camhpus club jodospace 交通学徐汇校区 徐家汇书院 上海气象博物馆 ROAD XUF Shanghai 徐汇区 DAD Expo Site 金 龙美术馆 上海世博园 电影汁博物馆 CHINAART MUSEUM 中华艺术宫 GUILIN ROAD 西岸美术馆 LONGHUA YUNJIN ROAD TEMPLE Kubrick书店 龙华寺 AVENUE"
text = "和平饭店"
prompt = get_poi_extraction_prompt_new(post_info=text)
print(prompt)
print("=========================================")
oai_call = OpenaiCall()
response = oai_call.chat(
    messages=[{"role": "user", "content": prompt}], model="gpt-3.5-turbo"
)
print(response)
