import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;

public class CombineInstances {

	HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();

	public CombineInstances(String path) {

		map.put("joy", new ArrayList<String>());
		map.put("shame", new ArrayList<String>());
		map.put("sadness", new ArrayList<String>());
		map.put("guilt", new ArrayList<String>());
		map.put("disgust", new ArrayList<String>());
		map.put("anger", new ArrayList<String>());
		map.put("fear", new ArrayList<String>());

		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line;
			while ((line = br.readLine()) != null) {
				System.out.println(line);

				Scanner scanner = new Scanner(line);
				scanner.useDelimiter("---");

				int id = scanner.nextInt();
				System.out.println("id = " + id);
				String sentiment = scanner.next();
				System.out.println("sentiment = " + sentiment);
				String instance = scanner.next();
				System.out.println("instance = " + instance);

				map.get(sentiment).add(instance);
			}

			System.out.println(map.get("joy").size());
			System.out.println(map.get("shame").size());
			System.out.println(map.get("sadness").size());
			System.out.println(map.get("guilt").size());
			System.out.println(map.get("disgust").size());
			System.out.println(map.get("anger").size());
			System.out.println(map.get("fear").size());

			br.close();
		} catch (Exception ex) {
			System.out
					.println("Exception in reading input file - combine instances");
			ex.printStackTrace();
		}
	}

	public void combineInstance(int numberOfInstances, String name) {
		try {
			PrintWriter out = new PrintWriter(new FileWriter("resources//"
					+ name + numberOfInstances));

			Iterator<Map.Entry<String, ArrayList<String>>> it = map.entrySet().iterator();

			int count = 1;
			
			while (it.hasNext()) {
				
				Map.Entry<String, ArrayList<String>> pairs = (Map.Entry<String, ArrayList<String>>)it.next();
				
				String sentiment = (String)pairs.getKey();
				ArrayList<String> joy = (ArrayList<String>)pairs.getValue();

				
				int i = 0;
				while (i < joy.size()) {
					int maxnumber = numberOfInstances <= (joy.size() - i) ? numberOfInstances
							: joy.size() - i;

					String aux = "";
					for (int j = 0; j < maxnumber; j++)
						aux += joy.get(i + j);

					out.println(count + "---"+sentiment+"---" + aux);

					count++;

					i += maxnumber;
				}

			}

			out.close();
		} catch (Exception ex) {
			System.out.println("Exception in combine instances - combine step");
			ex.printStackTrace();
		}

	}
}
