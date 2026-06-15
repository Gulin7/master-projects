# explain the function (paramters -> format) : result (format) + requirements (aka, AVG, ...)

from decimal import Decimal, getcontext

def get_deduped_records(records):
    # Handles empty input
    if len(records) == 0:
        return []

    unique = {}
    for record in records:
        found = False
        if record["sample_id"] in unique:
            # print(record)
            found = True
            if unique[record["sample_id"]]["timestamp"] < record["timestamp"]:
               unique[record["sample_id"]] = record
                  
        if not found:
            unique[record["sample_id"]] = record
            # print(unique[record["sample_id"]])

      # O(N)

    return unique.values()

"""
Implementation for IMP-4521: Deduplicate and Aggregate Sample Measurements
"""
def process_measurements(records):
   # Move deduplication logic to a separate function def get_deduped_records(records) -> returns list of deduped records
   # make unique a dictionary where key is sample_id and the value is the record entry (sample_id, timestamp, values)
   # make sure to compare timestamps and add the **latest** one (aka higher timestamp)
   #  unique = []
   #  for record in records:
   #      found = False
   #      for u in unique:
   #          if u["sample_id"] == record["sample_id"]:
   #              found = True
   #              break
   #      if not found:
   #          unique.append(record)

    unique = get_deduped_records(records)

    print(unique)

    result = []
    for u in unique:
        vals = u["values"]
        # Handles samples with empty values list (average = 0.0)

        # comment: extract average calculation into a separate function
        if len(vals) == 0:
            avg = 0
        else:
            total = 0
            for v in vals:
                total = total + v
                print(total)
            getcontext().prec = 2
            avg = float(Decimal(total) / Decimal(len(vals)))
            print(avg)
        result.append({"sample_id": u["sample_id"], "average": avg})

   # either use python sort() -> with sample_id as the key or write a SelectionSort function (maybe already exists in python)

   # O( N * (logN + M))

    sorted_results = sorted(result, key=lambda d: d["sample_id"])

    print(sorted_results)

   #  for i in range(len(result)):
   #      for j in range(i + 1, len(result)):
   #          if result[i]["sample_id"] > result[j]["sample_id"]:
   #              temp = result[i]
   #              result[i] = result[j]
   #              result[j] = temp

    return sorted_results

# add a unit test for the given requirements

def test_process_measurements():
    records = [
    {"sample_id": "B-002", "timestamp": 1001, "values": [3.0, 4.0]},
    {"sample_id": "A-001", "timestamp": 1000, "values": [1.5, 2.0, 2.5]},
    # newer, should be kept
    {"sample_id": "A-001", "timestamp": 1050, "values": [1.8, 2.2, 2.6]},
   ]
    
    processed_measurements = process_measurements(records)

    expected_results = [
    {"sample_id": "A-001", "average": 2.2},   # from values [1.8, 2.2, 2.6]
    {"sample_id": "B-002", "average": 3.5},   # from values [3.0, 4.0]
    ]

    assert(len(expected_results) == len(processed_measurements))

    for index in range(0, len(expected_results)):
      try:
         expected_id, id = expected_results[index]["sample_id"], processed_measurements[index]["sample_id"]

         print(f"Expected_id: {expected_id}, id: {id}")

         expected_average, average = expected_results[index]["average"], processed_measurements[index]["average"]
      except:
          print("Processed measurement fields not present")

      assert isinstance(expected_id, str)
      assert isinstance(id, str)
      assert isinstance(expected_average, float)
      assert isinstance(average, float)
      assert(expected_id == id)
      assert(expected_average == average)

def run_test_suites():
    test_process_measurements()

run_test_suites()