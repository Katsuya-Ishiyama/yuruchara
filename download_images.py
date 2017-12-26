# -*- coding: utf-8 -*-

import argparse
import csv
import os
import logging
import re
import sys
import time
import urllib
from bs4 import BeautifulSoup


MAX_PAGE = 40
SLEEP_TIME_SEC = 5
HTML_PARSER = 'lxml'
LOG_DIR = '/home/ishiyama/yuruchara'

logging.basicConfig(
    format='%(asctime)s %(message)s',
    filename=os.path.join(LOG_DIR, 'download_images.log'),
    level=logging.DEBUG
)


def check_character_type(character_type):
    if character_type not in ['gotochi', 'company']:
        raise ValueError('character_type must be "gotochi" or "company"')


class SourceDataUrlCollector(object):

    URL = 'http://www.yurugp.jp/vote/result_ranking.php?page={page}&sort={character_type_id}'
    IMAGE_URL_BASE = 'http://www.yurugp.jp/vote/{}'
    IMAGE_URL_PATTERN = re.compile(r'detail\.php\?id')
    IMAGE_URL_ID_PATTERN = re.compile(r'detail\.php\?id\=(\d{8})')
    CSS_TO_IMAGE_URL = '#charaList > ul.thumbnailList > li > a'
    SORT_ID_GOTOCHI = 1
    SORT_ID_COMPANY = 2

    def __init__(self, character_type):
        self.soup = None
        check_character_type(character_type)
        self.character_type = character_type

    def get_character_type_id(self):

        check_character_type(self.character_type)
        if self.character_type == 'gotochi':
            return self.SORT_ID_GOTOCHI
        else:
            return self.SORT_ID_COMPANY

    def get_source_url(self, page):
        self.src_url = self.URL.format(page=page, character_type_id=self.get_character_type_id())
        return self.src_url

    def collect(self, page):
        target_urls = []
        src_url = self.get_source_url(page)

        logging.info('SourceDataUrlCollector.download, {}'.format(src_url))

        with urllib.request.urlopen(src_url) as response:
            html = response.read()
            self.soup = BeautifulSoup(html, HTML_PARSER)
            detail_url_src = self.soup.select(self.CSS_TO_IMAGE_URL)
            for src in detail_url_src:
                url = src.get('href')
                logging.info('SourceDataUrlCollector.download: Current URL {}'.format(url))
                if self.IMAGE_URL_PATTERN.match(url):
                    logging.info('SourceDataUrlCollector.download: Include: {}'.format(url))
                    target_url = self.IMAGE_URL_BASE.format(url)
                    target_urls.append(target_url)
                    logging.info('SourceDataUrlCollector.download: Success: {}'.format(target_url))
                else:
                    logging.info('SourceDataUrlCollector.download: Exclude: {}'.format(url))

        if not target_url:
            logging.info('No more data.')
            raise ValueError('No more data.')

        return target_urls


class DataDownloader(object):

    CSS_RANK_POINT = '#detail > div.ttl_entry > ul > li.rank_pt'
    CSS_CHARACTER_NAME = '#detail > div.ttl_entry > h3'
    CSS_ENTRY_NO_PREFECTURE = '#detail > div.ttl_entry > ul > li.entry_no'
    CSS_MAIN_IMG = '#detail > div.mainImg > img'

    RANK_PATTERN = re.compile('^(\d+)位 ／ \d+pt$')
    POINT_PATTERN = re.compile('^\d+位 ／ (\d+)pt$')
    ENTRY_NO_PATTERN = re.compile(r'^エントリーNo.(\d+)（.+）$')
    PREFECTURE_PATTERN = re.compile(r'^エントリーNo.\d+（(.+)）$')

    def __init__(self, work_dir, csv_file, character_type):
        self.work_dir = work_dir
        check_character_type(character_type)
        self.character_type = character_type
        self.image_dir = os.path.join(work_dir, character_type, 'src', 'image')
        self.csv_file = csv_file
        self.urls = None
        self.fout = open(self.csv_file, 'w')
        self.writer = csv.writer(self.fout, lineterminator='\n')
        header = ['entry_no', 'character_type', 'character_name', 'prefecture', 'ranking',
                  'point', 'url', 'filename']
        self.writer.writerow(header)

    def extract_rank(self):
        rank_point_src = self.soup.select(self.CSS_RANK_POINT)[0].text
        ranking = self.RANK_PATTERN.findall(rank_point_src)[0]
        return ranking

    def extract_point(self):
        rank_point_src = self.soup.select(self.CSS_RANK_POINT)[0].text
        point = self.POINT_PATTERN.findall(rank_point_src)[0]
        return point

    def extract_character_name(self):
        character_name = self.soup.select(self.CSS_CHARACTER_NAME)[0].text
        return character_name

    def extract_entry_no(self):
        entry_no_prefecture_src = self.soup.select(self.CSS_ENTRY_NO_PREFECTURE)[0].text
        entry_no = self.ENTRY_NO_PATTERN.findall(entry_no_prefecture_src)[0]
        return entry_no

    def extract_prefecture(self):
        entry_no_prefecture_src = self.soup.select(self.CSS_ENTRY_NO_PREFECTURE)[0].text
        prefecture = self.PREFECTURE_PATTERN.findall(entry_no_prefecture_src)[0]
        return prefecture

    def download_image(self, output_dir):
        url_main_img = self.soup.select(self.CSS_MAIN_IMG)[0].get('src')
        filename = os.path.basename(url_main_img)
        urllib.request.urlretrieve(url_main_img, os.path.join(output_dir, filename))
        return (url_main_img, filename)

    def download(self, urls):

        if not isinstance(urls, list):
            raise ValueError('urls must be list.')

        self.urls = urls
        for url in self.urls:

            logging.info('DataDownloader.download, {}'.format(url))

            with urllib.request.urlopen(url) as response:
                html = response.read()
                self.soup = BeautifulSoup(html, HTML_PARSER)

            entry_no = self.extract_entry_no()
            character_type = self.character_type
            character_name = self.extract_character_name()
            prefecture = self.extract_prefecture()
            ranking = self.extract_rank()
            point = self.extract_point()
            url, filename = self.download_image(output_dir=self.image_dir)

            data = [entry_no, character_type, character_name, prefecture,
                    ranking, point, url, filename]
            self.writer.writerow(data)

            time.sleep(SLEEP_TIME_SEC)

    def close(self):
        self.fout.flush()
        self.fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str)
    parser.add_argument('character_type', type=str, choices=('gotochi', 'company'))
    args = parser.parse_args()
    work_dir = args.work_dir
    character_type = args.character_type

    source_url_collector = SourceDataUrlCollector(character_type=character_type)
    downloader = DataDownloader(
        work_dir=work_dir,
        csv_file=os.path.join(work_dir, character_type, 'yuruchara.csv'),
        character_type=character_type
    )

    for page in range(1, MAX_PAGE + 1):
        try:
            src_url = source_url_collector.collect(page)
        except:
            logging.info('Exit.')
            downloader.close()
            sys.exit(1)

        time.sleep(SLEEP_TIME_SEC)

        downloader.download(urls=src_url)

    downloader.close()
