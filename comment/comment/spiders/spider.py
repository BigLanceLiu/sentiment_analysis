from scrapy import Request,Spider
from lxml import etree
from comment.items import CommentItem


class CommentSpider(Spider):


    name = 'comment'
    def start_requests(self):
        templateurl = 'https://movie.douban.com/subject/26100958/?from=showing'
        for i in range(3000):
            url = templateurl.format(str(i * 1))
            print(url)
            yield Request(url=url, callback=self.parse)
    def parse(self, response):
        selector = etree.HTML(response.text)
        item = CommentItem()
        Comment_list=selector.xpath('//div[@class="comment"]/p/span/text()')
        for i in Comment_list:
            item['comments']=i
            yield item



    '''def get_movies(self):
        turl = 'http://video.mtime.com/search'
        yield Request(url=turl,callback=self.prase)

    def prase(self,response):
        selector = etree.HTML(response.text)
        Item = movieItem()
        movieItem_url = selector.xpath('//div[@class="xXnBC"]/dl/dd/h3/a/@href')
        for j in movieItem_url:
            movieItem['movies']=j
            yield movieItem

    '''
    '''
    def start_requests(self):
        templateurl = 'http://movie.mtime.com/87876/reviews/short/new-{}.html'
        for i in range(3000):
            url = templateurl.format(str(i * 1))
            print(url)
            yield Request(url=url, callback=self.parse)
    def parse(self, response):
        selector = etree.HTML(response.text)
        item = CommentItem()
        Comment_list=selector.xpath('//div[@class="mod_short"]/h3/text()')
        for i in Comment_list:
            item['comments']=i
            yield item
    '''
    '''
    def get_movies(self):
        turl = 'http://video.mtime.com/search'
        yield Request(url=turl,callback=self.prase_movies)

    def prase_movies(self,response):
        selector = etree.HTML(response.text)
        data = movieItem()
        movieItem_url = selector.xpath('//div[@class="xXnBC"]/dl/dd/h3/a/@href')
        for j in movieItem_url:
            data['movies']=j
            yield data

    '''

    '''
    def start_requests(self):
        templateurl = 'https://http://movie.mtime.com/262500/reviews/short/new-{}.html'
        for i in range(3000):
            url = templateurl.format(str(i * 20))
            print(url)
            yield Request(url=url, callback=self.parse)

    def parse(self, response):
        selector = etree.HTML(response.text)
        item = CommentItem()
        Comment_list=selector.xpath('//div[@class="comment"]/p/span/text()')
        for i in Comment_list:
            item['comments']=i
            yield item
    '''




