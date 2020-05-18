
import sentiment
from sentiment import sentiment

from flask import Flask, render_template, redirect, url_for

import gspread as gs
from oauth2client.service_account import ServiceAccountCredentials
from gspread.models import Cell

  


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('secret.json', scope)
client = gs.authorize(creds)

sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1AAvLhINPkyZUnbarXVBF5aOu9xyUCOXQEEVxvpzS2sc')

review_sheet = sheet.get_worksheet(0)
review_sheet_rank = sheet.get_worksheet(1)

review_list = review_sheet.get_all_records()
review_list_rank = review_sheet_rank.get_all_records()


app = Flask(__name__)

@app.route('/')
def hello():
    review_list_rank = review_sheet_rank.get_all_records()

    return render_template("home.html", items=review_list_rank)


#-----------------------------------------------------------------

@app.route('/train')
def train():
    sen = sentiment()
    for index in range(0,len(review_list)):
        x = sen.text_process(review_list[index]['Comment'])[0][0]
        review_sheet.update_cell(index+2, 7, x)

@app.route('/cal')
def cal():
    for index in range(0,len(review_list)):
        for index_rank in range(0,len(review_list_rank)):
            if review_list[index]['StoreName'] == review_list_rank[index_rank]['StoreName']:


                print('store :'  +review_list[index]['StoreName']+ ' rate :'+ str(review_list[index]['Rate'])+ ': count :'+ str(review_list_rank[index_rank]['Count']))

                review_list_rank[index_rank]['Rating'] += review_list[index]['Rate']

                if float(review_list[index]['Confidence'])  > 0.55555:
                    review_list_rank[index_rank]['Confidence'] = review_list_rank[index_rank]['Confidence'] + (review_list[index]['Confidence'] - 0.55555)
                else:
                    review_list_rank[index_rank]['Confidence'] = review_list_rank[index_rank]['Confidence'] - review_list[index]['Confidence']

                review_list_rank[index_rank]['Count'] += 1


    for index_rank in range(0,len(review_list_rank)):

        cells_rank = []
        cells_rank.append(Cell(row=index_rank+2, col=2, value=review_list_rank[index_rank]['Rating']))
        cells_rank.append(Cell(row=index_rank+2, col=3, value=review_list_rank[index_rank]['Confidence']))
        cells_rank.append(Cell(row=index_rank+2, col=4, value=review_list_rank[index_rank]['Count']))
                                  
        review_sheet_rank.update_cells(cells_rank)
    

    return redirect('/')


@app.route('/summery')
def summery():
    for index_rank in range(0,len(review_list_rank)):
        cells = []
        cells.append(Cell(row=index_rank+2, col=2, value=review_list_rank[index_rank]['Rating'] / review_list_rank[index_rank]['Count']))
        cells.append(Cell(row=index_rank+2, col=3, value=review_list_rank[index_rank]['Confidence'] / review_list_rank[index_rank]['Count']))

        review_sheet_rank.update_cells(cells)
        

    return redirect('/')


@app.route('/reset')
def reset():
    for index_rank in range(0,len(review_list_rank)):
        cells = []
        cells.append(Cell(row=index_rank+2, col=2, value=0))
        cells.append(Cell(row=index_rank+2, col=3, value=0))
        cells.append(Cell(row=index_rank+2, col=4, value=0))

        review_sheet_rank.update_cells(cells)

    return redirect('/')


@app.route('/D/<int:id>')
def show_detail(id):
    item_comment = []
    item_store = []

    for index_rank in range(0,len(review_list_rank)):
        if id == review_list_rank[index_rank]['ID']:
            item_store = review_list_rank[index_rank]


    for index in range(0,len(review_list)):
        if review_list[index]['StoreName'] == item_store['StoreName']:
            item_comment.append(review_list[index])

    return render_template("page.html", store=item_store, items=item_comment)


if __name__ == '__main__':
    app.run('localhost', 4449)