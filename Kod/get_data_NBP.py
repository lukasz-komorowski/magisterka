import requests
import json
import objectpath
import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv


# Funkcja pobierająca dane z API NBP
def get_rates(start_date, end_date, currency):
    # pobieranie odpowiedzie dla określonego okresu i określonej waluty
    response = requests.get(f'https://api.nbp.pl/api/exchangerates/rates/a/{currency}/{start_date}/{end_date}/?format=json', timeout=10)
    # 'dobieranie się' do danych
    data = json.loads(response.text)
    json_tree = objectpath.Tree(data['rates'])
    # tuple z kursami i datami
    rate_tuple = tuple(json_tree.execute('$..mid'))
    date_tuple = tuple(json_tree.execute('$..effectiveDate'))
    # listy z kursami i datami
    rate_list = list(rate_tuple)
    date_list = list(date_tuple)
    response.close()
    return rate_list, date_list

# arbitralnie duża liczba dni - przekraczająca zasięg czasowy API (02.01.2002)
days_number = 8000
# dzielenie dni na okresy po 200
period_length = 200
iterations = days_number//period_length
# lista list kursów i dat (surowe dane)
rates_raw = []
dates_raw = []
# data końcowa i początkowa okresu pobierania
end_date = datetime.datetime.now()
start_date = datetime.datetime.now() - datetime.timedelta(days=200)
# pętla pobierająca
for i in range(iterations):
    # wykorzystanie funkcji do pobierania danych
    tmp_rates, tmp_dates = get_rates(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), "chf")
    # kumulowanie pobranych okresów razem
    rates_raw.append(tmp_rates)
    dates_raw.append(tmp_dates)
    # print(i, ". ", start_date.strftime("%Y-%m-%d"), " do ", end_date.strftime("%Y-%m-%d"))
    # aktualizacja dat (przesuwamy je o 200+1 dni
    start_date = start_date - datetime.timedelta(days=period_length+1)
    end_date = end_date - datetime.timedelta(days=period_length+1)
    # Jeżeli przekroczony zakres API --> zakończ działanie
    if(end_date < datetime.datetime(2002, 1, 2)):
        break

# listy wyczyszczonych danych - kursów i dat
rates_list = []
dates_list = []
# czyszczenie danych: lista list --> lista
for tmp in rates_raw:
    rates_list = tmp + rates_list
for tmp in dates_raw:
    dates_list = tmp + dates_list

# łączenie danych w pary (data, kurs)
dates_rates_list = list(zip(dates_list, rates_list))

dates_rates_list_titeled = [("Data", "Kurs")] + dates_rates_list
print(dates_rates_list_titeled)
# zapis danych do pliku CSV na potrzeby dalszej pracy bez konieczności ciągłego pobierania danych od nowa
file_name = 'NBP_dane.csv'
file = open(file_name, 'w+', newline='')
with file:
    write = csv.writer(file)
    write.writerows(dates_rates_list_titeled)


# #drukowanie wykresu walut (CHF/PLN)
# x = np.linspace(-len(rates_list) + 1, 0, len(rates_list))
# plt.plot(x, rates_list, label="CHF/PLN")
# plt.xlabel('Dni (bez weekendów)')
# plt.ylabel('Kurs')
# plt.title('Kurs franka szwajcarskiego')
# plt.legend()
# plt.show()

