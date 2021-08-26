#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-24 17:14:26
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-24 17:51:33
FilePath: /Chinese-Text-Classification/dermatology/dermatology/spiders/dermatology.py
Description: 
'''
import scrapy
from scrapy.http import Request
from scrapy import Selector
from selenium import webdriver
import datetime
import re

class DermatologySpider(scrapy.Spider):
    name = 'dermatology'
    allowed_domains = ['www.haodf.com']

    def __init__(self, *args, **kwargs):
        super(DermatologySpider, self).__init__(*args, **kwargs)
        self.page = 100
        self.start_urls = ['https://www.haodf.com/faculty/DE4rO-XCoLU0Jpnmi88QpNK6eR/zixun.htm', 
                           'https://www.haodf.com/faculty/DE4rO-XCoLUnCoaEJefaGY-Fi8/zixun.htm',
                           'https://www.haodf.com/faculty/DE4r08xQdKSLewmj9gzM21bXlzuR/zixun.htm',
                           'https://www.haodf.com/faculty/DE4rO-XCoLUOy6tCObWl99uMTa/zixun.htm']
        self.option = webdriver.ChromeOptions()
        self.option.add_argument('headless')
        self.option.add_argument('no=sandbox')
        self.option.add_argument('blink-setting=imagesEnable=false')
        self.driver = None
    
    def start_requests(self):
        for url in self.start_urls:
            yield Request(url=url, callback=self.parse)

    def parse(self, response):
        self.driver = webdriver.Chrome(executable_path='/Volumes/yyx/projects/Chinese-Text-Classification/dermatology/dermatology/spiders/chromedriver',
                                       chrome_options=self.option)
        self.driver.set_page_load_timeout(3000)
        self.driver.get(response.url)

        for i in range(self.page):
            while not self.driver.find_element_by_xpath("//*[@id='gray']/div[5]/div[1]/div[3]/div[1]/div/div[3]/div/a[8]").text:  # 没有找到下一页
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 模拟窗口向下滑动
            texts = self.driver.find_elements_by_xpath("//*[@id='adv_list_det']/li/a")
            for i in range(len(texts)):
                line = {"label": "皮肤科", "text": texts[i].text}
                print(line)
                yield line
            try:
                self.driver.find_element_by_xpath("//*[@id='gray']/div[5]/div[1]/div[3]/div[1]/div/div[3]/div/a[8]").click()  # 模拟点击下一页
            except Exception as e:
                print(e)
                print("*"*20)
                break

