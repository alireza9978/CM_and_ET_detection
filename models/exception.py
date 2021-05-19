class HourlyDateException(Exception):
    def __init__(self, message="Error happened in hourly data's date conversion",
                 persian_message="در تبدیل تاریخ داده های ساعتی مشکلی وجود داشت"):
        self.message = message
        self.persian_message = persian_message


class MonthlyDateException(Exception):
    def __init__(self, message="Error happened in monthly data's date conversion",
                 persian_message="در تبدیل تاریخ داده های ماهانه مشکلی وجود داشت"):
        self.message = message
        self.persian_message = persian_message

class WrongColumnsException(Exception):
    def __init__(self, message="The given dataframe has the wrong number of columns",
                 persian_message="تعداد ستون های دیتاست ورودی اشتباه می باشد"):
        self.message = message
        self.persian_message = persian_message

class WrongDateFormatException(Exception):
    def __init__(self, message="Wrong format of data in Date column",
                 persian_message="فرمت ستون تاریخ در دیتاست اشتباه است"):
        self.message = message
        self.persian_message = persian_message
