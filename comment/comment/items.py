# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class CommentItem(scrapy.Item):
    comments = scrapy.Field()

'''class movieItem(scrapy.Item):
    movies = scrapy.Field()
'''

