#include <complex>
#include <sys/timeb.h>
#include <map>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <ppl.h>
#include <filesystem>
#include <string>

#include "fftw3.h"
#pragma comment(lib,"libfftw3-3.lib")

using namespace std;
namespace fs = std::experimental::filesystem;
using namespace concurrency;

#define REAL 0
#define IMAG 1

#define CHUNK_SIZE 128
#define OFFSET_SIZE 16

typedef struct WaveHeader {
	char chunk_id[4] = { 0 };//big
	unsigned int chunk_size = 0;//little
	char format[4] = { 0 };//big
	char fmt_chunk_id[4] = { 0 };//big
	unsigned int fmt_chunk_size = 0;//little
	unsigned short audio_fomat = 0;//little
	unsigned short num_channels = 0;//little
	unsigned int sample_rate = 0;//little
	unsigned int byte_rate = 0;//little
	unsigned short block_align = 0;//little
	unsigned short bits_per_sample = 0;//little
	char data_chunk_id[4] = { 0 };//big
	unsigned int data_chunk_size = 0;//little
	int num_frame = 0;
	int start_pos = 0;
};

void PrintFileInfo(WaveHeader &wave_header) {
	cout << "\n================================" << endl;
	cout << "chunk_size      " << wave_header.chunk_size << endl;
	cout << "format          " << wave_header.format << endl;
	cout << "fmt_chunk_size  " << wave_header.fmt_chunk_size << endl;
	cout << "audio_fomat     " << wave_header.audio_fomat << endl;
	cout << "num_channels    " << wave_header.num_channels << endl;
	cout << "sample_rate     " << wave_header.sample_rate << endl;
	cout << "byte_rate       " << wave_header.byte_rate << endl;
	cout << "block_align     " << wave_header.block_align << endl;
	cout << "bits_per_sample " << wave_header.bits_per_sample << endl;
	cout << "data_chunk_size " << wave_header.data_chunk_size << endl;
	cout << "num_frame       " << wave_header.num_frame << endl;
	cout << "start_pos       " << wave_header.start_pos << endl;
	cout << "================================" << endl;
}


const int HEAD_LENGTH = 1 * 1024 * 1024;//1M
char buf[HEAD_LENGTH];

void getHead(string fname, WaveHeader &wh) {
	FILE *stream;
	freopen_s(&stream, fname.c_str(), "rb", stderr);
	fread(buf, 1, HEAD_LENGTH, stream);
	int pos = 0;
	while (pos < HEAD_LENGTH) {
		if (buf[pos] == 'R'&&buf[pos + 1] == 'I'&&buf[pos + 2] == 'F'&buf[pos + 3] == 'F') {
			wh.chunk_id[0] = 'R';
			wh.chunk_id[1] = 'I';
			wh.chunk_id[2] = 'F';
			wh.chunk_id[3] = 'F';
			pos += 4;
			break;
		}
		++pos;
	}
	wh.chunk_size = *(int *)&buf[pos];
	pos += 4;
	wh.format[0] = buf[pos];
	wh.format[1] = buf[pos + 1];
	wh.format[2] = buf[pos + 2];
	wh.format[3] = buf[pos + 3];
	pos += 4;
	while (pos < HEAD_LENGTH) {
		if (buf[pos] == 'f'&&buf[pos + 1] == 'm'&&buf[pos + 2] == 't') {
			wh.fmt_chunk_id[0] = 'f';
			wh.fmt_chunk_id[1] = 'm';
			wh.fmt_chunk_id[2] = 't';
			pos += 4;
			break;
		}
		++pos;
	}
	wh.fmt_chunk_size = *(int *)&buf[pos];
	pos += 4;
	wh.audio_fomat = *(short *)&buf[pos];
	pos += 2;
	wh.num_channels = *(short *)&buf[pos];
	pos += 2;
	wh.sample_rate = *(int *)&buf[pos];
	pos += 4;
	wh.byte_rate = *(int *)&buf[pos];
	pos += 4;
	wh.block_align = *(short *)&buf[pos];
	pos += 2;
	wh.bits_per_sample = *(short *)&buf[pos];
	pos += 2;
	while (pos < HEAD_LENGTH) {
		if (buf[pos] == 'd'&&buf[pos + 1] == 'a'&&buf[pos + 2] == 't'&buf[pos + 3] == 'a') {
			wh.data_chunk_id[0] = 'd';
			wh.data_chunk_id[1] = 'a';
			wh.data_chunk_id[2] = 't';
			wh.data_chunk_id[3] = 'a';
			pos += 4;
			break;
		}
		++pos;
	}
	wh.data_chunk_size = *(int *)&buf[pos];
	pos += 4;
	wh.start_pos = pos;
	wh.num_frame = wh.data_chunk_size / (wh.num_channels*(wh.bits_per_sample / 8));
	PrintFileInfo(wh);
}


fftw_complex x[CHUNK_SIZE] = { 0 };
fftw_complex y[CHUNK_SIZE] = { 0 };
fftw_plan plan = fftw_plan_dft_1d(CHUNK_SIZE, x, y, FFTW_FORWARD, FFTW_ESTIMATE);

void getMap(string fname, map<size_t, int> &data_map, WaveHeader &wh) {
	int pos = wh.start_pos;

	FILE *stream;
	freopen_s(&stream, fname.c_str(), "rb", stderr);
	char* file_data = new char[wh.chunk_size + 8];
	fread(file_data, 1, wh.chunk_size + 8, stream);

	hash<double> hash;
	double max = 0;
	double dist = 0;
	int index = 0;
	double temp = 0;
	int pos_gap = wh.num_channels * wh.bits_per_sample / 8;

	//新的方法，大幅减少读取次数
	//每次滑动时memcpy重叠部分，只需要读取OFFSET_SIZE大小的新数据
	/*for (; index < CHUNK_SIZE; index++) {
		x[index][REAL] = *(short*)&file_data[pos];
		pos += pos_gap;
	}
	fftw_execute(plan);
	for (int j = CHUNK_SIZE / 8; j < CHUNK_SIZE / 4; j++) {
		temp = y[j][REAL] * y[j][REAL] + y[j][IMAG] * y[j][IMAG];
		if (temp > max) {
			max = temp;
			dist = j;
		}
	}
	data_map[max + dist] = pos;
	int fftw_complex_size = sizeof(fftw_complex);
	for (; pos < wh.start_pos + wh.data_chunk_size;) {
		memcpy(x, x + OFFSET_SIZE, (CHUNK_SIZE - OFFSET_SIZE) * fftw_complex_size);
		index = CHUNK_SIZE - OFFSET_SIZE;
		for (; index < CHUNK_SIZE; index++) {
			x[index][REAL] = *(short*)&file_data[pos];
			pos += pos_gap;
		}
		fftw_execute(plan);
		max = 0;
		dist = 0;
		for (int j = CHUNK_SIZE / 8; j < CHUNK_SIZE / 4; j++) {
			temp = y[j][REAL] * y[j][REAL] + y[j][IMAG] * y[j][IMAG];
			if (temp > max) {
				max = temp;
				dist = j;
			}
		}
		data_map[max + dist] = pos - CHUNK_SIZE*pos_gap;
	}*/

	//原来的方法
	for (int i = pos; i < wh.start_pos + wh.data_chunk_size;) {
		x[index][REAL] = 0.5*(*(short*)&file_data[i] + *(short*)&file_data[i + 2]);
		//x[index][REAL] = *(short*)&file_data[i];
		++index;
		if (index == CHUNK_SIZE) {
			index = 0;
			fftw_execute(plan);
			max = 0;
			dist = 0;
			for (int j = CHUNK_SIZE / 8; j < CHUNK_SIZE / 4; j++) {
				temp = y[j][REAL] * y[j][REAL] + y[j][IMAG] * y[j][IMAG];
				if (temp > max) {
					max = temp;
					dist = j;
				}
			}

			i -= (CHUNK_SIZE - OFFSET_SIZE) * pos_gap;
			//TODO:添加哈希及哈希冲突解决!
			data_map[max + dist] = i;
		}
		else {
			i += pos_gap;
		}
	}

	cout << "\nmap size:  " << data_map.size() << endl;
	delete[] file_data;
	file_data = NULL;
}

void checkMap(map<size_t, int> map_audio, map<size_t, int> &map_sample) {
	map<int, int> check_map;
	map<size_t, int>::iterator itr_find;
	size_t related_pos = 0;
	for (map<size_t, int>::iterator itr = map_sample.begin(), end = map_sample.end(); itr != end; ++itr) {
		itr_find = map_audio.find(itr->first);
		if (itr_find != map_audio.end()) {
			related_pos = itr_find->second - itr->second;
			++check_map[related_pos];
		}
	}

	int max_value = 0;
	cout << "check map size : " << check_map.size() << endl;
	for (map<int, int>::iterator itr = check_map.begin(), end = check_map.end(); itr != end; ++itr) {
		max_value = itr->second > max_value ? itr->second : max_value;
	}
	cout << "max_value: " << max_value << endl;
	double matching_rate = (double)max_value / map_sample.size();
	cout << "matching rate: " << matching_rate << endl;
	check_map.clear();
}

int main() {
	struct timeb startTime, endTime;
	ftime(&startTime);
	srand(time(NULL));

	//库里的音频
	string f_audio = "..\\audio\\audio.wav";
	WaveHeader wh_audio;
	map<size_t, int> map_audio;
	getHead(f_audio, wh_audio);
	getMap(f_audio, map_audio, wh_audio);

	//待检测样本所在的文件夹
	string path_sample = "..\\sample";
	vector<string> f_samples;
	for (auto & p : fs::directory_iterator(path_sample)) {
		f_samples.push_back(p.path().string());
	}

	int num_file_sample = f_samples.size();
	vector<WaveHeader> wh_samples(num_file_sample);
	vector<map<size_t, int>> map_samples(num_file_sample);

	for (int i = 0; i < num_file_sample; i++) {
		getHead(f_samples[i], wh_samples[i]);
	}

	//每个文件开一个线程建库以及匹配
	task_group tg;
	for (int index = 0; index < num_file_sample; index++) {
		tg.run([&map_audio, &map_samples, &f_samples, &wh_samples, index]() {getMap(f_samples[index], map_samples[index], wh_samples[index]); checkMap(map_audio, map_samples[index]); });
	}
	tg.wait();


	ftime(&endTime);
	cout << "\n总时间：" << (endTime.time - startTime.time) + (endTime.millitm - startTime.millitm) / 1000.0 << "秒" << endl;

	cout << "\nPress Enter to exit..." << endl;
	cin.get();

	return 0;
}