# -*- coding: utf-8 -*-
"""
This script is used to get data from TripAdvisor.

Usage:
    scrapy runspider ./scripts/get_data.py -o ./data/mount_etna_data.csv -t csv

"""
##################
# Importing libraries
##################

import click
from scrapy.item import Field
from scrapy.item import Item
from scrapy.spiders import Spider
from scrapy.selector import Selector
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose, TakeFirst
import re
from typing import Union

import utility
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

##################
# Constants
##################
# Links to read
START_URLS = utility.links

# Absolute xpath
COMMENTS_BLOCK_XPATH = '//*[@id="REVIEWS"]/section/div[1]/div/div[5]/div'

# Relative xpaths
AUTHOR_XPATH = ".//span/div/div[1]/div[1]/div[2]/span/a/text()"
PROFILE_URL_XPATH = ".//div[1]/a/@href"
PLACE_XPATH = ".//span/div/div[1]/div[1]/div[2]/div/div/span[1]/text()"
CONTRIBUTIONS_XPATH = ".//span/div/div[1]/div[1]/div[2]/div/div/span[2]/text()"
TITLE_XPATH = ".//span/div/div[4]/a/span/text()"
CONTENT_XPATH = ".//div[contains(@style,'-webkit-line-clamp')]/div/span/text()"
DATE_XPATH = ".//span/div/div[8]/div[1]/text()"
RATING_XPATH = './/span/div/div[3]/svg/@title'

##################
# Functions to parse
##################


def parse_profile_url(url: str) -> str:
    return "https://www.tripadvisor.com" + url


def parse_rating(rating_title: str) -> int:
    return int(rating_title[0])


def parse_place(place_content: str) -> Union[str, None]:
    return None if "contrib" in place_content.lower() else place_content


def parse_date(date_text: str) -> str:
    date_text = date_text.replace('Written ', '')
    return date_text


def parse_contributions(contributions_text: str) -> int:
    return int(''.join(re.findall(r"\d", contributions_text)))


##################
# Getting data and saving it
##################


class Comment(Item):
    author = Field()
    title = Field()
    content = Field()
    place = Field(input_processor=MapCompose(parse_place))
    date = Field(input_processor=MapCompose(parse_date))
    rating = Field(input_processor=MapCompose(parse_rating),
                   output_processor=TakeFirst())
    profile_url = Field(input_processor=MapCompose(parse_profile_url),
                        output_processor=TakeFirst())
    url_path = Field()
    contributions = Field(input_processor=MapCompose(parse_contributions),
                          output_processor=TakeFirst())


class TripAdvisorSpider(Spider):
    name = "Trip Spider"
    start_urls = START_URLS

    def parse(self, response):
        sel = Selector(response)
        comments = sel.xpath(COMMENTS_BLOCK_XPATH)
        comments = comments[1:-1]
        for elem in comments:
            item = ItemLoader(Comment(), elem)
            # add_xpath(<field>, <xpath>)
            item.add_xpath('author', AUTHOR_XPATH)
            item.add_xpath('title', TITLE_XPATH)
            item.add_xpath('content', CONTENT_XPATH)
            item.add_xpath('place', PLACE_XPATH)
            item.add_xpath('date', DATE_XPATH)
            item.add_xpath('rating', RATING_XPATH)
            item.add_xpath('profile_url', PROFILE_URL_XPATH)
            item.add_value('url_path', response.url)
            item.add_xpath('contributions', CONTRIBUTIONS_XPATH)

            yield item.load_item()

        next_pages = response.css('.cCnaz > div:nth-child(1) > a:nth-child(1)')

        if len(next_pages) > 0:
            for next_page in next_pages:
                yield response.follow(next_page, self.parse)
