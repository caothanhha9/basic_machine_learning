# -*- coding: utf-8 -*-


bid_amount = input("Bạn muốn trả tôi bao nhiêu tiền (x 1000 vnd) cho bài học này?")

if float(bid_amount) >= 100.0:
    print(':D ồ ồ, cảm ơn bạn. Bạn thật hào phóng.')
else:
    print('cảm ơn bạn đã thanh toán dịch vụ.')