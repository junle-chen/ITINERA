import requests
import re
from bs4 import BeautifulSoup
from functools import partial
import json
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from .utils import text_contains, class_starts_with, class_contains

# https://michael-shub.github.io/curl2scrapy/
headers = {
    "authority": "edith.xiaohongshu.com",
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
    "origin": "https://www.xiaohongshu.com",
    "referer": "https://www.xiaohongshu.com/",
    "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "x-b3-traceid": "8f7e9cb6b20eb2b4",
    "x-s": "XYW_eyJzaWduU3ZuIjoiNTEiLCJzaWduVHlwZSI6IngxIiwiYXBwSWQiOiJ4aHM\
tcGMtd2ViIiwic2lnblZlcnNpb24iOiIxIiwicGF5bG9hZCI6IjE0YTg1NWRkMTVjY2RlODZlOG\
U2YTk2NGZlMjMyNjg2ZjE4NjhhZmZjN2IwYWU0N2QxN2U2ODExYTZiZDQwMDg3YWM2MmZhMWNhZD\
QxYjVjNjEwYjNmOGE1MjM2ZWYzZmM5ZTNiZmRhMWZhYTFlYjkwZDc0YWEzMWI1NGM3MmNkMGQ3NG\
FhMzFiNTRjNzJjZGFjNDg5YjlkYThjZTVlNDhmNGFmYjlhY2ZjM2VhMjZmZTBiMjY2YTZiNGNjM2N\
iNTlmYjYyODAzMTM3MjIyNmJjN2RkNGJiNGU3NjZiMmNhZmVhMzhkYzc5MWNiYWM3YjdlYzBlYzM1\
ZGQxZTFlNTIxYjQxM2IwNWM1YmY2ZDM0Y2IwODMxODhiZGQyMjgzYzJjYTc1OTllZWQxOTM2ZmFjYT\
AxMjk4OTE1ZjlhNzliZWIyNjZhN2E5YzBkNzkzZjVhYjgzYjgyYmFhODZhOWJhMzY2NjU3ZDgzM\
2Q0YTU2ZWY5ZDFjNmZiMzhkMjE5OSJ9",
    "x-s-common": "2UQAPsHCPUIjqArjwjHjNsQhPsHCH0rjNsQhPaHCH0P1PUhAHjIj2eHjwjQ+GnPW/M\
PjNsQhPUHCHdYiqUMIGUM78nHjNsQh+sHCH0H1P/r1+UHVHdWMH0ijP/WhGAPI80+jP/LI4g8Y+nYYP98IqgSl\
JBTk8nMD+gQY+e8j+oSiPdPAPeZIPeHAw/Z9PaHVHdW9H0il+0W7w/cl+/LhPeWMNsQh+UHCHSY8pMRS2LkCGp4\
D4pLAndpQyfRk/SzbyLleadkYp9zMpDYV4Mk/a/8QJf4hanS7ypSGcd4/pMbk/9St+BbH/gz0zFMF8eQnyLSk49S\
0Pfl1GflyJB+1/dmjP0zk/9SQ2rSk49S0zFGMGDqEybkea/8QyDLInpzdPLEgLfT+pb8xn/QaJrRrnflOzMLUnpz3\
PDEonfl+yDME/fkdPSkxz/zwyfYinfMyyDhUag48pMLI/0Qz2rhUp/QOzrphnpzyypkrLg4+zBqAnp4+PDMTnfY+pF\
EinDzz2bSxpfkwyDp7nnkwJLRoz/b+yDFUnS482SkT//pyprEknfMayrMgnfY8pr8Vnnk34MkrGAm8pFpC/p4QPLEo//\
++JLE3/L4zPFEozfY+2D8k/SzayDECafkyzF8x/Dzd+pSxJBT8pBYxnSznJrEryBMwzF8TnnkVybDUnfk+PS8i/nkyJ\
pkLcfS+ySDUnpzyyLEo/fk+PDEk/SzVJpSxngSOzrbC/pz+PFMxagSwJLkx/0QayFEoafSwzMLA/fkyyLMT/fYyJp8i\
/gkiyMSCGAp+pFEknp4+PMSx8Bl82DQVngk+PpkoLgYypr8V/SzQ2bSxLgY+PDS7/S4+PpSTn/QyzrFIn/QQ4FRr/\
gYOzBYknD4z2LMx87k82Dkxnpz0PLRLJBlypMbh/Mz+PSkTzfk8prbh/nk3+rRLz/byyfli/dkVypkgagSwySki/\
0Qb+pSCcfTw2fTCnfknybSx87k8yf4EnnMByrRrnfYOpFki/gk8PDExp/+yzB4C//QzPbSLp/QypMDMnDzByDETnfS\
+2fY3/nkb+LR/a0DjNsQhwsHCHDDAwoQH8B4AyfRI8FS98g+Dpd4daLP3JFSb/BMsn0pSPM87nrldzSzQ2bPAGdb7z\
gQB8nph8emSy9E0cgk+zSS1qgzianYt8p+1/LzN4gzaa/+NqMS6qS4HLozoqfQnPbZEp98QyaRSp9P98pSl4oSzcg\
mca/P78nTTL0bz/sVManD9q9z1J7+xJMcM2gbFnobl4MSUcdb6agW3tF4ryaRApdz3agWIq7YM47HFqgzkanTU4FSk\
N7+3G9PAaL+P8DDA/9LI4gzVP0mrnd+P+nprLFkSyS87PrSk8nphpd4PtMmFJ7Ql4BYcpFTS2bDhJFSeL0bQzgQ/8M8\
7cD4l4bQQ2rL68LzD8p8M49kQcAmAPgbFJDS3qrTQyrzA8nLAqMSDLe80p/pAngbF2fbr8Bpf2drl2fc68p4gzjTQ2o8S\
LM8FNFSba9pDLocEqdkMpLR6pD4Q4f4SygbF4aR889phydbTanTP4FSkzbmoGnMxag8iJaTQweYQygkMcS87JrS9zFGF8g8\
SzbP78/bM4r+QcA4AzBPROaHVHdWEH0iTP0q9+AqIweZINsQhP/Zjw08R",
    "x-t": "1687941558085",
}

headers_ios = {
    "authority": "edith.xiaohongshu.com",
    "accept": "application/json, text/plain, */*",
    "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
    "origin": "https://www.xiaohongshu.com",
    "referer": "https://www.xiaohongshu.com/",
    "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/84.0.4147.122 Mobile/15E148 Safari/604.1",
    "x-b3-traceid": "8f7e9cb6b20eb2b4",
    "x-s": "XYW_eyJzaWduU3ZuIjoiNTEiLCJzaWduVHlwZSI6IngxIiwiYXBwSWQiOiJ4aHM\
tcGMtd2ViIiwic2lnblZlcnNpb24iOiIxIiwicGF5bG9hZCI6IjE0YTg1NWRkMTVjY2RlODZlOG\
U2YTk2NGZlMjMyNjg2ZjE4NjhhZmZjN2IwYWU0N2QxN2U2ODExYTZiZDQwMDg3YWM2MmZhMWNhZD\
QxYjVjNjEwYjNmOGE1MjM2ZWYzZmM5ZTNiZmRhMWZhYTFlYjkwZDc0YWEzMWI1NGM3MmNkMGQ3NG\
FhMzFiNTRjNzJjZGFjNDg5YjlkYThjZTVlNDhmNGFmYjlhY2ZjM2VhMjZmZTBiMjY2YTZiNGNjM2N\
iNTlmYjYyODAzMTM3MjIyNmJjN2RkNGJiNGU3NjZiMmNhZmVhMzhkYzc5MWNiYWM3YjdlYzBlYzM1\
ZGQxZTFlNTIxYjQxM2IwNWM1YmY2ZDM0Y2IwODMxODhiZGQyMjgzYzJjYTc1OTllZWQxOTM2ZmFjYT\
AxMjk4OTE1ZjlhNzliZWIyNjZhN2E5YzBkNzkzZjVhYjgzYjgyYmFhODZhOWJhMzY2NjU3ZDgzM\
2Q0YTU2ZWY5ZDFjNmZiMzhkMjE5OSJ9",
    "x-s-common": "2UQAPsHCPUIjqArjwjHjNsQhPsHCH0rjNsQhPaHCH0P1PUhAHjIj2eHjwjQ+GnPW/M\
PjNsQhPUHCHdYiqUMIGUM78nHjNsQh+sHCH0H1P/r1+UHVHdWMH0ijP/WhGAPI80+jP/LI4g8Y+nYYP98IqgSl\
JBTk8nMD+gQY+e8j+oSiPdPAPeZIPeHAw/Z9PaHVHdW9H0il+0W7w/cl+/LhPeWMNsQh+UHCHSY8pMRS2LkCGp4\
D4pLAndpQyfRk/SzbyLleadkYp9zMpDYV4Mk/a/8QJf4hanS7ypSGcd4/pMbk/9St+BbH/gz0zFMF8eQnyLSk49S\
0Pfl1GflyJB+1/dmjP0zk/9SQ2rSk49S0zFGMGDqEybkea/8QyDLInpzdPLEgLfT+pb8xn/QaJrRrnflOzMLUnpz3\
PDEonfl+yDME/fkdPSkxz/zwyfYinfMyyDhUag48pMLI/0Qz2rhUp/QOzrphnpzyypkrLg4+zBqAnp4+PDMTnfY+pF\
EinDzz2bSxpfkwyDp7nnkwJLRoz/b+yDFUnS482SkT//pyprEknfMayrMgnfY8pr8Vnnk34MkrGAm8pFpC/p4QPLEo//\
++JLE3/L4zPFEozfY+2D8k/SzayDECafkyzF8x/Dzd+pSxJBT8pBYxnSznJrEryBMwzF8TnnkVybDUnfk+PS8i/nkyJ\
pkLcfS+ySDUnpzyyLEo/fk+PDEk/SzVJpSxngSOzrbC/pz+PFMxagSwJLkx/0QayFEoafSwzMLA/fkyyLMT/fYyJp8i\
/gkiyMSCGAp+pFEknp4+PMSx8Bl82DQVngk+PpkoLgYypr8V/SzQ2bSxLgY+PDS7/S4+PpSTn/QyzrFIn/QQ4FRr/\
gYOzBYknD4z2LMx87k82Dkxnpz0PLRLJBlypMbh/Mz+PSkTzfk8prbh/nk3+rRLz/byyfli/dkVypkgagSwySki/\
0Qb+pSCcfTw2fTCnfknybSx87k8yf4EnnMByrRrnfYOpFki/gk8PDExp/+yzB4C//QzPbSLp/QypMDMnDzByDETnfS\
+2fY3/nkb+LR/a0DjNsQhwsHCHDDAwoQH8B4AyfRI8FS98g+Dpd4daLP3JFSb/BMsn0pSPM87nrldzSzQ2bPAGdb7z\
gQB8nph8emSy9E0cgk+zSS1qgzianYt8p+1/LzN4gzaa/+NqMS6qS4HLozoqfQnPbZEp98QyaRSp9P98pSl4oSzcg\
mca/P78nTTL0bz/sVManD9q9z1J7+xJMcM2gbFnobl4MSUcdb6agW3tF4ryaRApdz3agWIq7YM47HFqgzkanTU4FSk\
N7+3G9PAaL+P8DDA/9LI4gzVP0mrnd+P+nprLFkSyS87PrSk8nphpd4PtMmFJ7Ql4BYcpFTS2bDhJFSeL0bQzgQ/8M8\
7cD4l4bQQ2rL68LzD8p8M49kQcAmAPgbFJDS3qrTQyrzA8nLAqMSDLe80p/pAngbF2fbr8Bpf2drl2fc68p4gzjTQ2o8S\
LM8FNFSba9pDLocEqdkMpLR6pD4Q4f4SygbF4aR889phydbTanTP4FSkzbmoGnMxag8iJaTQweYQygkMcS87JrS9zFGF8g8\
SzbP78/bM4r+QcA4AzBPROaHVHdWEH0iTP0q9+AqIweZINsQhP/Zjw08R",
    "x-t": "1687941558085",
}

cookies = {
    "a1": "188c30f3b150uva5ha3fpqyqlkiemd5ra46b4yh2s30000239061",
    "webId": "7b3ef57181f484cbd9767558c54b442a",
    "gid": "yYYSq8iqi40JyYYSq8iqD4vxy287h02x0qiUKJMAvA1T3Dq8d6f2F4888Jqj8Ky8DiJy0j8y",
    "gid.sign": "WAFuLJ9fLM/MW4L5ofdqxXFPM6U=",
    "webBuild": "2.11.7",
    "xsecappid": "xhs-pc-web",
    "web_session": "030037a3a8bbf037e7dbc8a9e3234a91320b14",
    "websectiga": "634d3ad75ffb42a2ade2c5e1705a73c845837578aeb31ba0e442d75c648da36a",
    "sec_poison_id": "7971d651-c530-420d-b3c9-da774f9c06aa",
}


class XHSScraper:
    def __init__(self):
        pass

    def extract_post_content(self, url):
        # ⚡ 优化：并行发送两个HTTP请求
        import time
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(requests.get, url, headers=headers, cookies=cookies)
            future2 = executor.submit(requests.get, url, headers=headers_ios, cookies=cookies)
            
            resp = future1.result()
            resp2 = future2.result()
        
        print(f"⏱️  Both HTTP requests completed in {time.time() - start_time:.2f}s")
        
        if resp.status_code != 200:
            raise ValueError(
                f"HTTP response failed with status code {resp.status_code}"
            )
        if resp2.status_code != 200:
            raise ValueError(
                f"HTTP response IOS failed with status code {resp2.status_code}"
            )

        soup = BeautifulSoup(resp.content, "lxml")
        title = soup.find("meta", attrs={"name": "og:title"})["content"]

        meta_content = soup.find("meta", attrs={"name": "description"}).get("content")

        ori_url = soup.find("meta", attrs={"name": "og:url"})["content"]

        scripts = soup.find(
            partial(text_contains, substr="imageList", tag_name="script")
        ).text

        imgs = re.findall(r'"imageList":\[(.*)\]', scripts)
        imgs = imgs[0] if imgs else ""
        len_imgs = len(imgs)
        imgs = imgs.encode("UTF-8").decode("unicode_escape")
        imgs = re.findall(r'"(https://.*?)"', imgs) + re.findall(
            r'"(http://.*?)"', imgs
        )

        final_urls_map = {}  # 使用字典来确保唯一性

        for url in imgs:
            if "avatar" in url:
                continue

            # 提取文件ID - 处理图片URL（包含!符号的）
            file_id_match = re.search(r"/([^/]+)!", url)

            if file_id_match:
                # 图片URL - 提取!之前的ID部分
                file_id = file_id_match.group(1)

                # 优先保留 'nd_dft_' (default) 版本，其次是 'nd_prv_' (preview)
                if file_id not in final_urls_map or "nd_dft_" in url:
                    final_urls_map[file_id] = url
            else:
                # 视频URL或其他 - 提取最后一段路径作为ID
                # 例如: 01e8fb8adc7fb1984f0370019a16978f72_258.mp4
                video_id_match = re.search(r"/([^/]+\.(mp4|mov|m4v))$", url)
                if video_id_match:
                    video_id = video_id_match.group(1)
                    # 视频URL去重 - 只保留第一个遇到的
                    if video_id not in final_urls_map:
                        final_urls_map[video_id] = url
                else:
                    # 其他情况，使用完整URL作为key
                    if url not in final_urls_map:
                        final_urls_map[url] = url

        # 5. 循环处理最终的、干净的 URL 列表
        image_urls, video_urls = [], []
        for raw_url in final_urls_map.values():
            if "avatar" in raw_url:
                continue
            
            base_url = raw_url.split("?")[0]

            if base_url.lower().endswith((".mp4", ".mov", ".m4v")):
                video_urls.append(base_url)
            else:
                image_urls.append(raw_url + "?imageView2/format/jpg|imageMogr2/strip")
        
        if len(image_urls) == 1 and len(video_urls) != 0:
            # 处理只有一个图片但有视频的情况，可能是视频封面图
            image_urls = []
        elif image_urls:
            # 避免live变成视频
            video_urls = []
        

        tags_str = soup.find_all(partial(class_contains, substr="tag"))
        if tags_str:
            tags = [t.text for t in tags_str]
        else:
            tags = None

        regs = [
            r'"user":(\{"avatar":(.*?)\})',
            r'"user":(\{"nickname":(.*?)\})',
            r'"user":(\{"userId":(.*?)\})',
        ]
        user = [item for reg in regs for item in re.findall(reg, scripts)]
        if user:
            user = json.loads(user[0][0])
            nickname = user["nickname"]
            avatar = user["avatar"]
            userId = user["userId"]
        else:
            nickname = None
            avatar = None
            userId = None

        interact = re.findall(r'interactInfo":(\{(.*?)\})', scripts)

        if interact:
            interact = json.loads(interact[0][0])
            collectedCount = interact["collectedCount"]
            commentCount = interact["commentCount"]
            shareCount = interact["shareCount"]
            likedCount = interact["likedCount"]
        else:
            collectedCount = None
            commentCount = None
            shareCount = None
            likedCount = None

        date_str = soup.find(attrs={"class": "date"}).text.split()
        date = date_str[0]
        if len(date_str) > 1:
            city = date_str[1]
        else:
            city = None

        # ⚡ 第二个响应已经在开始时并行获取了
        # resp2 = requests.get(url, headers=headers_ios, cookies=cookies)  # 删除这行
        # if resp2.status_code != 200:
        #     raise ValueError(
        #         f"HTTP response IOS failed with status code {resp2.status_code}"
        #     )
        soup2 = BeautifulSoup(resp2.content, "lxml")

        keywords_tags = soup2.find("meta", attrs={"name": "keywords"})
        if keywords_tags:
            keywords = keywords_tags["content"].split(",")
        else:
            keywords = None

        pois_str = soup2.find_all(partial(class_contains, substr="note-poi"))
        if pois_str:
            poi = pois_str[0].text.strip()
        else:
            poi = None

        scripts2 = soup2.find(
            partial(text_contains, substr="commentData", tag_name="script")
        )
        if not scripts2:
            comments = None
            return {
                "source_platform": "xhs",
                "url": ori_url,
                "title": title,
                "content": meta_content,
                "img_urls": image_urls,
                "video_urls": video_urls,
                "author": {
                    "nickname": nickname,
                    "avatar": avatar,
                    "userId": userId,
                    "city": city,
                },
                "liked_count": likedCount,
                "collected_count": collectedCount,
                "comment_count": commentCount,
                "share_count": shareCount,
                "comments": comments,
                "time": date,
                "tags": tags,
                "keywords": keywords,
                "location": poi,
            }
        scripts2 = scripts2.text

        raw_comments = re.findall(
            r'commentData":(\{(.*?)\}),"userOtherNotesData"', scripts2
        )
        if raw_comments:
            raw_comments = json.loads(raw_comments[0][0])
            # TODO: subcomments?
            comments = [
                {
                    "content": com["content"],
                    "user": com["user"]["nickname"],
                    "is_from_author": com["user"]["userId"] == userId,
                }
                for com in raw_comments["comments"]
            ]
        else:
            comments = None

        return {
            "source_platform": "xhs",
            "url": ori_url,
            "title": title,
            "content": meta_content,
            "img_urls": image_urls,
            "video_urls": video_urls,
            "author": {
                "nickname": nickname,
                "avatar": avatar,
                "userId": userId,
                "city": city,
            },
            "liked_count": likedCount,
            "collected_count": collectedCount,
            "comment_count": commentCount,
            "share_count": shareCount,
            "comments": comments,
            "time": date,
            "tags": tags,
            "keywords": keywords,
            "location": poi,
        }


if __name__ == "__main__":
    url = "https://www.xiaohongshu.com/explore/67dd8119000000001d02f76d?xsec_token=ABSO6S-bEcyA44-o3ESfto_kYILTRAz6QseFGnMcfzWY0=&xsec_source=pc_search&source=unknown"
    url = "https://www.xiaohongshu.com/explore/67dd8119000000001d02f76d?xsec_token=ABSO6S-bEcyA44-o3ESfto_kYILTRAz6QseFGnMcfzWY0=&xsec_source=pc_search&source=unknown"
    url = "https://www.xiaohongshu.com/explore/6914432e00000000040163cc?xsec_token=ABWYFw-e7-KP8vnNLbrHBZmPZq1bEPcLLnKRuhXDUzuYY=&xsec_source=pc_feed"
    url = "https://www.xiaohongshu.com/explore/690c58e1000000000703a6da?xsec_token=ABZh8YQ0ggh77QBoEDpVJlFAi8wmc7CPXvVVu1UkL-GKc=&xsec_source=pc_feed"
    url = "https://www.xiaohongshu.com/explore/6910a9fd000000000700e3bb?xsec_token=ABJub241xr7cjhi3TRYCMSydqFzxOEVpJDbGEvN3QMpIc=&xsec_source=pc_feed"
    url = "https://www.xiaohongshu.com/explore/6826b55a000000002300d355?xsec_token=ABm7BYr7pBQblNgOUehE4PQCIDdAJ6BvXTGNuXGp_JSp8=&xsec_source=pc_search&source=unknown"
    url = "https://www.xiaohongshu.com/explore/6904c3bd000000000503936f?xsec_token=ABMN_wQTMr6kO20vh2HJjHEwbH2dqFaxqOIPapz3qrRGM=&xsec_source=pc_feed"
    url = "https://www.xiaohongshu.com/explore/690ed9b3000000000700cac8?xsec_token=ABrq2WtzwWEhPFsLKB10W35_l_KapvqBPLKjNeuras2nA=&xsec_source=pc_search&source=unknown"
    # url = "https://www.xiaohongshu.com/explore/64fac943000000001f03bb4f"
    resp = XHSScraper().extract_post_content(url)

    pprint(resp)
    print(len(resp["img_urls"]))
