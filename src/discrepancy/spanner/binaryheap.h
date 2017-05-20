#ifndef BINHEAP_H
#define BINHEAP_H


#include <cstdio>
#include <vector>
#include <map>
#include <limits>

static const unsigned int MAXUINT = std::numeric_limits<unsigned int>::max();

template <class TValue>
class FixedArrayLookup {
private:
	TValue* lookup;
	unsigned int mySize;
	TValue myDefault;
public:
	FixedArrayLookup() {
	}
	FixedArrayLookup(unsigned int size, TValue myDefault) {
		construct(size, myDefault);
	}
	FixedArrayLookup(const FixedArrayLookup &other) {
		construct(other.mySize, other.myDefault);
		for (unsigned int i = 0; i < mySize; i++)
			lookup[i] = other.lookup[i];
	}
	~FixedArrayLookup() {
		delete[] lookup;
	}
	TValue& operator[](const unsigned int index) {
		return lookup[index];
	}
	void construct(unsigned int size, TValue tdefault) {
		mySize = size;
		myDefault = tdefault;
		lookup = new TValue[size];
		clear();
	}
	unsigned int getSize() {
		return mySize;
	}
	void clear() {
		for (unsigned int i = 0; i < mySize; i++)
			lookup[i] = myDefault;
	}
};

template <class TValue>
class MapLookup {
private:
	std::map<unsigned int, TValue> lookup;
	TValue myDefault;
public:
	MapLookup() {
	}
	MapLookup(const int size, TValue myDefault) : myDefault(myDefault) {
	}
	MapLookup(const MapLookup &other) : lookup(other.lookup) {
		myDefault = other.myDefault;
	}
	TValue& operator[](const unsigned int& index) {
		typename std::map<unsigned int, TValue>::iterator it = lookup.find(index);
		if (it == lookup.end()) {
			//const TValue myDefault2 = myDefault;
			//lookup.insert(it, myDefault2);
			lookup[index] = myDefault;
			return lookup.find(index)->second;
		}
		return it->second;
	}
	void construct(unsigned int /*size*/, TValue myDefault) {
		this->myDefault = myDefault;
	}
	unsigned int getSize() {
		return lookup.size();
	}
	void clear() {
		lookup.clear();
	}
};

template <class TValue, template<class> class Lookup = FixedArrayLookup, bool decreaseKeyInserts = false>
class BinHeap
{
private:
    inline unsigned int leftIndex(const unsigned int i) const {
        return (i << 1) + 1;
    }
    inline unsigned int rightIndex(const unsigned int i) const {
        return (i << 1) + 2;
    }
    inline unsigned int parentIndex(const unsigned int i) const {
        return (i - 1) >> 1;
    }
	inline void swap(const unsigned int i, const unsigned int j) {
		lookup[values[i].first] = j;
		lookup[values[j].first] = i;
		std::pair<unsigned int, TValue> temp = values[i];
		values[i] = values[j];
		values[j] = temp;
	}
	void decreaseKeyFix(unsigned int i) {
		while (i > 0 && values[parentIndex(i)].second > values[i].second) {
			swap(i, parentIndex(i));
			i = parentIndex(i);
		}
	}
	void minHeapify(unsigned int i) {
		unsigned int smallest;
		while (1) {
			if (leftIndex(i) < myCount && values[leftIndex(i)].second < values[i].second)
				smallest = leftIndex(i);
			else
				smallest = i;
			if (rightIndex(i) < myCount && values[rightIndex(i)].second < values[smallest].second)
				smallest = rightIndex(i);
			if (smallest != i) {
				swap(i, smallest);
				i = smallest;
			}
			else
				break;
		}
	}
public:
	TValue myNonexistingValue;
	unsigned int myCount;
	std::pair<unsigned int, TValue> defPair;
	Lookup<std::pair<unsigned int, TValue> > values;
	Lookup<unsigned int> lookup;
	BinHeap() : myCount(0), values(), lookup() {
	}
	BinHeap(unsigned int size, TValue nonexistingValue) 
    : myNonexistingValue(nonexistingValue), myCount(0), values(size, defPair), lookup(size, MAXUINT), defPair(0, nonexistingValue) {
	}
	BinHeap(const BinHeap<TValue, Lookup, true> &other) 
    : myNonexistingValue(other.myNonexistingValue), myCount(other.myCount), values(other.values), lookup(other.lookup), defPair(0, other.myNonexistingValue) {
	}
	BinHeap(const BinHeap<TValue, Lookup, false> &other) 
    : myNonexistingValue(other.myNonexistingValue), myCount(other.myCount), values(other.values), lookup(other.lookup), defPair(0, other.myNonexistingValue) {
	}
	void construct(unsigned int size, TValue nonexistingValue) {
		myNonexistingValue = nonexistingValue;
		lookup.construct(size, MAXUINT);
		values.construct(size, defPair);
	}
	void clear(bool clearLookup = true) {
		myCount = 0;
		if (clearLookup)
			lookup.clear();
	}
	void insert(const unsigned int key, const TValue value) {
		std::pair<unsigned int, TValue> pair(key, value);
		lookup[key] = myCount;
		values[myCount] = pair;
		decreaseKeyFix(myCount);
		myCount++;
	}
	inline TValue getValue(const unsigned int key) {
		if (!contains(key)) return myNonexistingValue;
		return values[lookup[key]].second;
	}
	inline std::pair<unsigned int, TValue> getMin() {
		return values[0];
	}
	void remove(const unsigned int key) {
		myCount--;
		values[lookup[key]] = values[myCount];
		lookup[values[myCount].first] = lookup[key];
		minHeapify(lookup[key]);
		lookup[key] = MAXUINT;
	}
	void decreaseKey(const unsigned int key, const TValue newValue) {
		if (contains(key)) {
			values[lookup[key]].second = newValue;
			decreaseKeyFix(lookup[key]);
			//remove(key);
			//insert(key, newValue);
		}
		else if (decreaseKeyInserts) {
			insert(key, newValue);
		}
	}
	void increaseKey(const unsigned int key, const TValue newValue) {
		values[lookup[key]].second = newValue;
		minHeapify(lookup[key]);
	}
	void extractMin() {
		myCount--;
		unsigned int key = values[0].first;
		values[0] = values[myCount];
		lookup[values[myCount].first] = 0;
		minHeapify(0);
		lookup[key] = MAXUINT;
	}
	inline unsigned int getCount() const {
		return myCount;
	}
	inline bool contains(const unsigned int key) {
		return lookup[key] != MAXUINT;
	}
};

#endif // BINHEAP_H
