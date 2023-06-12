import copy
import datetime
import random
import traceback
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from .poetry import BeamGenerator, GreedyGenerator_GAN

def save_texts_to_file(texts, out_file):
    with open(out_file, 'w', encoding="utf-8") as outf:
        outf.write('\n'.join(texts))

def get_params_number(model):
    return sum(t.numel() for t in model.parameters())

def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))

def train_loop(model, train_dataset, val_dataset, criterion, save_path=None,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    early_optimizer_SD = None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    need_to_save=True,
                    need_to_gen=False,
                    need_long=True,
                    tokenizer=None,
                    phrase="Война - это"):
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке и производится тестовая генерация предложения.
    :param model: torch.nn.Module - обучаемая модель
    :param train_dataset: torch.utils.data.Dataset - данные для обучения
    :param val_dataset: torch.utils.data.Dataset - данные для валидации
    :param criterion: функция потерь
    :param save_path: при необходимости сохранять модель на каждой удачной эпохе вводится путь до папки сохранения (по умолчанию None)
    :param lr: скорость обучения
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param early_stopping_patience: наибольшее количество эпох, в течение которых допускается отсутствие улучшения модели, чтобы обучение продолжалось
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи (по умолчанию torch.utils.data.DataLoader)
    :param optimizer_ctor: оптимизатор (по умолчанию Adam)
    :param early_optimizer_SD: для продолжения обучения необходимы также веса для оптимизатора, что загрузятся через load_state_dict (по умолчанию None)
    :param need_to_save: если модель необходимо сохранить на каждой удачной эпохе, ключ ставится на True (по умолчанию False)
    :param need_to_gen: ключ для тестовой генерации предложения после каждой эпохи (по умолчанию True)
    :param need_long: ключ, если для работы библиотеки необходим формат torch.tensor.long (по умолчанию True)
    :param tokenizer: токинизатор для генерации фразы
    :param pharse: фраза, продолжение которой будет генерироваться
    :return: кортеж из трёх элементов:
        - история среднего значения функции потерь на валидации
        - лучшая модель
        - оптимизатор
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)
    
    if early_optimizer_SD is not None:
        optimizer.load_state_dict(early_optimizer_SD)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)
    loss_history = []

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print(f'Эпоха {epoch_i}')
            
            #-----Тренировка модели-----
            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):                
                # Пакеты данных переносятся в нужном формате на нужный device
                if need_long:
                    batch_x = torch.tensor(batch_x).to(torch.long)
                    batch_y = torch.tensor(batch_y).to(torch.long)
                else:
                    batch_x = torch.tensor(batch_x)
                    batch_y = torch.tensor(batch_y)

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                # Делаем предсказание следующих токенов, считаем функцию потерь
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                # Градиентный спуск
                model.zero_grad()
                loss.backward()
                optimizer.step()

                # Служебное
                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            #-----Валидация модели-----
            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):                    
                    # Пакеты данных переносятся в нужном формате на нужный device
                    if need_long:
                        batch_x = torch.tensor(batch_x).to(torch.long)
                        batch_y = torch.tensor(batch_y).to(torch.long)
                    else:
                        batch_x = torch.tensor(batch_x)
                        batch_y = torch.tensor(batch_y)

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    # Делаем предсказание следующих токенов, считаем функцию потерь
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    # Служебное
                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            loss_history.append(mean_val_loss)

            # Если модель оказалась лучше, чем предыдущая лучшая
            if mean_val_loss < best_val_loss:
                # Сохраняем модель
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)

                # Если need_to_save=True, то сохраняем эту модель ещё и на диск
                if need_to_save:
                    torch.save({
                        'model_state_dict': best_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss
                    }, 
                    save_path + f"model_{str(best_epoch_i)}.tar")
                
                print('Новая лучшая модель!')

                # Производим тестовую генерацию предложения
                if need_to_gen and tokenizer is not None:
                    beam_generator = BeamGenerator(best_model, tokenizer)
                    beam_gen_variants = beam_generator(phrase, beamsize=5, return_hypotheses_n=1)
                    for score, pred_txt in beam_gen_variants:
                        print(pred_txt)

            # Если модель долго не улучшается, значит, она достигла порога
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break
    
    return loss_history, best_model, optimizer

def train_loop_GAN(model_D, model_G, dataset, criterion,
                    lr_D=1e-4, lr_G=1e-4, epoch_n=10, batch_size=32,
                    device=None, l2_reg_alpha=0,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    early_optimizer_SD_D = None,
                    early_optimizer_SD_G = None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    need_to_gen=False,
                    tokenizer=None,
                    phrase="Война - это",
                    end_chunk=100):
    """
    Цикл для обучения моделей в генеративно-состязательной сети. После каждой эпохи производится тестовая генерация предложения.
    :param model_D: torch.nn.Module - обучаемая модель дискиминатора
    :param model_G: torch.nn.Module - обучаемая модель генератора
    :param dataset: torch.utils.data.Dataset - данные для обучения
    :param criterion: функция потерь для настройки модели
    :param lr_D: скорость обучения дискриминатора
    :param lr_G: скорость обучения генератора
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи (по умолчанию torch.utils.data.DataLoader)
    :param optimizer_ctor: оптимизатор (по умолчанию Adam)
    :param early_optimizer_SD_D: для продолжения обучения необходимы также веса для оптимизатора дискриминатора, что загрузятся через load_state_dict (по умолчанию None)
    :param early_optimizer_SD_G: для продолжения обучения необходимы также веса для оптимизатора генератора, что загрузятся через load_state_dict (по умолчанию None)
    :param need_to_save: если модель необходимо сохранить на каждой удачной эпохе, ключ ставится на True (по умолчанию False)
    :param need_to_gen: ключ для тестовой генерации предложения после каждой эпохи (по умолчанию True)
    :param tokenizer: токинизатор для генерации фразы
    :param pharse: фраза, продолжение которой будет генерироваться
    :param end_chunk: параметр максимальной длины сгенерированной фразы
    :return: кортеж из шести элементов:
        - модель дискриминатора
        - модель оптимизатора
        - оптимизатор дискриминатора
        - оптимизатор генератора
        - история среднего значения функции потерь дискриминатора
        - история среднего значения функции потерь генератора
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model_D.to(device)
    model_G.to(device)

    if optimizer_ctor is None:
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr_D, weight_decay=l2_reg_alpha)
        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr_G, weight_decay=l2_reg_alpha)
    else:
        optimizer_D = optimizer_ctor(model_D.parameters(), lr=lr_D)
        optimizer_G = optimizer_ctor(model_G.parameters(), lr=lr_G)
    
    if early_optimizer_SD_D is not None and early_optimizer_SD_G is not None:
        optimizer_D.load_state_dict(early_optimizer_SD_D)
        optimizer_G.load_state_dict(early_optimizer_SD_G)

    text_dataloader = data_loader_ctor(dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)

    loss_history_D = []
    loss_history_G = []

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print(f'Эпоха {epoch_i}')

            model_D.train()
            model_G.train()

            mean_loss_D = 0
            mean_loss_G = 0
            train_batches_n = 0

            for batch_i, (start_text, end_text) in tqdm(enumerate(text_dataloader)):                
                # -----Тренировка дискриминатора-----
                model_D.zero_grad()

                # Берём текст
                start_text = torch.tensor(start_text)
                end_text = torch.tensor(end_text)

                start_text = copy_data_to_device(start_text, device)
                end_text = copy_data_to_device(end_text, device)

                # Тренировка с настоящими предложениями
                # Создаём единицы размерности batch
                batch_size = start_text.size(0)
                label = torch.full((batch_size, ), 1, dtype=torch.float, device=device)

                # Прогоняем конец текста через дискриминатор
                output = model_D(end_text).view(-1)

                # Вычисляем loss реальных предложений
                err_D_real = criterion(output, label)
                err_D_real.backward()
                
                # Тренировка cо сгенерированными предложениями
                # Прогоняем начало предложений через дискриминатор
                gen_model = GreedyGenerator_GAN(model_G)
                pred_txt = gen_model(start_text, max_steps_n=end_chunk)

                # Создаём нули размерности batch
                label.fill_(0)

                # Прогоняем через дискриминатор
                output = model_D(pred_txt).view(-1)

                # Вычисляем loss сгенерированных предложений
                err_D_fake = criterion(output, label)
                err_D_fake.backward()

                # Вычисляем общий loss
                err_D = err_D_real + err_D_fake

                # Обновляем оптимизатор для дискиминатора
                optimizer_D.step()

                # -----Тренировка генератора-----
                model_G.zero_grad()

                # Создаём единицы размерности batch 
                # 1 - это достижение для тренировки генератора, что он смог обмануть дискриминатор
                label.fill_(1)

                # Прогоняем через дискриминатор ранее сгенерированные предложения (end_text_gen)
                output = model_D(pred_txt).view(-1)

                # Вычисляем loss
                err_G = criterion(output, label)
                err_G.backward()

                # Обновляем оптимизатор для генератора
                optimizer_G.step()

                #-----Служебное-----
                train_batches_n += 1
                mean_loss_D += float(err_D)
                mean_loss_G += float(err_G)


            print(f"Эпоха прошла за {np.round((datetime.datetime.now() - epoch_start).total_seconds(), 2)} секунд")
            mean_loss_D /= train_batches_n
            mean_loss_G /= train_batches_n
            print(f"Средняя функция потерь для дискриминатора = {mean_loss_D}")
            loss_history_D.append(mean_loss_D)
            print(f"Средняя функция потерь для генератора = {mean_loss_G}")
            loss_history_G.append(mean_loss_G)

            # Тестовая генерация предложения после эпохи
            if need_to_gen and tokenizer is not None:
                gen_model = BeamGenerator(model_G, tokenizer)
                beam_gen_variants = gen_model(phrase, beamsize=5, return_hypotheses_n=1)
                for score, pred_txt in beam_gen_variants:
                    print(pred_txt)

            print("Следующая эпоха")
        
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break
    
    return model_D, model_G, optimizer_D, optimizer_G, loss_history_D, loss_history_G
