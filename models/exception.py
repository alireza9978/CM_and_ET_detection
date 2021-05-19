class HourlyDateException(Exception):
    def __init__(self, message="Error happened in hourly data's date conversion", persian_message = "در تبدیل تاریخ داده های ساعتی مشکلی وجود داشت"):
        self.message = message
        self.persian_message = persian_message

class MonthlyDateException(Exception):
    def __init__(self, message="Error happened in monthly data's date conversion", persian_message = "در تبدیل تاریخ داده های ماهانه مشکلی وجود داشت"):
        self.message = message
        self.persian_message = persian_message

